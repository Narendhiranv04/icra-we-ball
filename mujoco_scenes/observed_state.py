"""Persistent observed-object registry, graph, events, and stage visualizations."""

from __future__ import annotations

import json
import hashlib
import csv
import math
import os
import re
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.geometry_checker import (
    CUMULATIVE_VISUALIZATION_PURPOSE,
    GeometryChecker,
    MeasurementEvidence,
    PointCloudRun,
    read_ply,
    voxel_downsample,
    write_ply,
)
from mujoco_scenes.geometry_properties import (
    extract_object_properties,
    load_geometry_config,
    pairwise_relation_evaluation,
)
from mujoco_scenes.task_witness import (
    DEFAULT_TASK_REQUIREMENTS_PATH,
    evaluate_geometric_requirements,
    evaluate_joint_task_witness,
    evaluate_semantic_compatibility,
    evaluate_task_witness,
    evaluate_usage_policy_task_witness,
    build_target_compatibility_matrix,
    load_task_requirements,
    PAIRING_STRATEGIES,
    resolve_task_requirements_path,
)
from mujoco_scenes.semantic_grounding import (
    NullSemanticDetector,
    SemanticDetector,
    load_semantic_config,
    run_semantic_inspection,
)


SCHEMA_VERSION = 3
STATUS_COLORS = {
    "previous": (61, 116, 184),
    "new": (45, 166, 85),
    "updated": (230, 126, 34),
}
RELATION_COLORS = {
    "TRUE": (45, 166, 85),
    "FALSE": (210, 62, 62),
    "UNKNOWN": (135, 135, 135),
    "OBSERVED": (45, 85, 140),
}
EVIDENCE_COLORS = (
    (49, 103, 189),
    (230, 126, 34),
    (45, 166, 85),
    (139, 92, 246),
    (14, 148, 160),
    (205, 66, 82),
    (126, 132, 36),
    (190, 74, 161),
)


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _evidence_color(object_id: str) -> tuple[int, int, int]:
    """Return a stable, visibly distinct colour for a generic object ID."""
    match = re.search(r"(\d+)$", object_id)
    index = int(match.group(1)) - 1 if match else 0
    return EVIDENCE_COLORS[index % len(EVIDENCE_COLORS)]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_compatibility_csv(path: Path, matrix: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    cells = matrix.get("cells", [])
    fieldnames = sorted(
        {key for cell in cells for key in cell}
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow(cell)
    temporary.replace(path)


def _atomic_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write_ply(temporary, points, colors)
    temporary.replace(path)


def _safe_run_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("run_id must contain at least one letter or digit")
    return safe


def _property_number(record: dict[str, Any] | None) -> float | None:
    if not record or record.get("value") is None:
        return None
    try:
        value = float(record["value"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _dimension_values(record: dict[str, Any]) -> list[float | None]:
    dimensions = record.get("dimensions_m", {})
    return [
        _property_number(dimensions.get(name))
        for name in ("length", "width", "height")
    ]


def _combined_candidate_status(*statuses: str) -> str:
    if "FALSE" in statuses:
        return "FALSE"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    return "TRUE"


def _semantic_quality_key(record: dict[str, Any] | None) -> tuple:
    """Deterministically rank cached semantic observations."""
    if not isinstance(record, dict):
        return (0, 0, -1.0, -1.0)
    quality = record.get("quality", {})
    return (
        1 if record.get("status") == "SUPPORTED" else 0,
        int(quality.get("supporting_view_count") or 0),
        float(quality.get("mean_confidence") or -1.0),
        float(quality.get("winning_label_margin") or -1.0),
    )


def _select_validated_semantic(
    previous: dict[str, Any] | None,
    observation: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    """Retain a stronger cached semantic result after weak re-observation."""
    if (
        observation.get("status") == "SUPPORTED"
        and _semantic_quality_key(observation)
        >= _semantic_quality_key(previous)
    ):
        return deepcopy(observation), True
    return deepcopy(previous), False


class ObservedStateRun:
    """One persistent registry and growing graph across observation stages."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        scene_name: str,
        region_ids: Iterable[str],
        initial_region_id: str = "countertop",
        voxel_size: float = 0.003,
        geometry_config: str | Path | None = None,
        task_requirements: str | Path | dict[str, Any] | None = None,
        semantic_detector: SemanticDetector | None = None,
        semantic_config: str | Path | dict[str, Any] | None = None,
        grounding_mode: str = "geometry-only",
        pairing_strategy: str | None = None,
        save_semantic_overlays: bool = False,
        record_oracle_diagnostics: bool = True,
        run_config: dict[str, Any] | None = None,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_dir.name
        self.scene_name = scene_name
        self.initial_region_id = initial_region_id
        self.voxel_size = voxel_size
        self.config = (
            load_geometry_config(geometry_config)
            if geometry_config
            else load_geometry_config()
        )
        self.task_requirements = load_task_requirements(task_requirements)
        joint_task = self.task_requirements.get("_task_schema") in {
            "JOINT_ROLE_GROUNDING",
            "JOINT_USAGE_POLICY_GROUNDING",
        }
        configured_pairing_strategy = self.task_requirements.get(
            "pairing", {}
        ).get(
            "strategy",
            "semantic_role_scoped" if joint_task else "exhaustive_all_pairs",
        )
        self.pairing_strategy = (
            pairing_strategy or configured_pairing_strategy
        ).replace("-", "_")
        if self.pairing_strategy not in PAIRING_STRATEGIES:
            raise ValueError(
                "pairing_strategy must be one of "
                f"{sorted(PAIRING_STRATEGIES)}"
            )
        self.grounding_mode = grounding_mode
        self.semantic_detector = semantic_detector or NullSemanticDetector()
        self.semantic_enabled = not isinstance(
            self.semantic_detector, NullSemanticDetector
        )
        if pairing_strategy is None and not self.semantic_enabled:
            # Semantic scoping is impossible without semantic evidence.
            # Geometry-only diagnostics therefore retain exhaustive pairing.
            self.pairing_strategy = "exhaustive_all_pairs"
        if isinstance(semantic_config, dict):
            self.semantic_config = deepcopy(semantic_config)
        else:
            self.semantic_config = load_semantic_config(
                semantic_config
                if semantic_config is not None
                else Path(__file__).resolve().parent
                / "configs"
                / "semantic_grounding.yaml"
            )
        self.save_semantic_overlays = bool(save_semantic_overlays)
        self.record_oracle_diagnostics = bool(record_oracle_diagnostics)
        if isinstance(task_requirements, dict):
            task_source = "inline"
        else:
            task_source = str(
                resolve_task_requirements_path(
                    task_requirements or DEFAULT_TASK_REQUIREMENTS_PATH
                )
            )
        self.registry_path = self.run_dir / "object_registry.json"
        self.graph_path = self.run_dir / "observed_graph.json"
        self.latest_witness_path = self.run_dir / "latest_witness.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.stages_dir = self.run_dir / "stages"
        self.stages_dir.mkdir(exist_ok=True)
        self.region_ids = tuple(region_ids)

        if self.registry_path.exists():
            self.registry = json.loads(self.registry_path.read_text())
            if self.registry.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Run {self.run_dir} uses observed-state schema "
                    f"{self.registry.get('schema_version')}; region-evidence "
                    f"schema {SCHEMA_VERSION} requires a new --run-id"
                )
        else:
            self.registry = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "scene_name": scene_name,
                "voxel_size_m": voxel_size,
                "current_stage": -1,
                "instance_index": {},
                "objects": {},
            }
        self.next_stage = int(self.registry.get("current_stage", -1)) + 1
        self.latest_witness = (
            json.loads(self.latest_witness_path.read_text())
            if self.latest_witness_path.exists()
            else None
        )
        config_payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "scene_name": scene_name,
            "created_at": datetime.now().astimezone().isoformat(),
            "voxel_size_m": voxel_size,
            "regions": [self.initial_region_id, *self.region_ids],
            "geometry_config": str(
                Path(geometry_config).resolve()
                if geometry_config
                else "configs/geometry_inference.yaml"
            ),
            "inspection_rig_config": "configs/inspection_rigs.yaml",
            "property_measurement_source": (
                "STAGE_LOCAL_REGION_GATED_MEASUREMENT_EVIDENCE"
            ),
            "cumulative_cloud_purpose": (
                CUMULATIVE_VISUALIZATION_PURPOSE
            ),
            "inference_basis": (
                grounding_mode.upper().replace("-", "_")
            ),
            "grounding_mode": grounding_mode,
            "pairing_strategy": self.pairing_strategy,
            "semantic_detector_enabled": self.semantic_enabled,
            "record_oracle_diagnostics": self.record_oracle_diagnostics,
            "task_requirements": task_source,
            "task_id": self.task_requirements["task_id"],
            **(run_config or {}),
        }
        if not (self.run_dir / "run_config.json").exists():
            _atomic_json(self.run_dir / "run_config.json", config_payload)
        goal_instruction = self.task_requirements.get("goal_instruction")
        if goal_instruction:
            _atomic_json(
                self.run_dir / "goal_instruction.json",
                {
                    "task_id": self.task_requirements["task_id"],
                    "goal_instruction": goal_instruction,
                    "specification_source": self.task_requirements.get(
                        "specification_source"
                    ),
                    "generated_from_foundation_model": bool(
                        self.task_requirements.get(
                            "generated_from_foundation_model", False
                        )
                    ),
                },
            )
        _atomic_json(
            self.run_dir / "task_requirements.json",
            {
                key: deepcopy(value)
                for key, value in self.task_requirements.items()
                if not key.startswith("_")
            },
        )

    @classmethod
    def create_for_scene(
        cls,
        scene,
        *,
        runs_root: str | Path = "runs",
        run_id: str | None = None,
        voxel_size: float = 0.003,
        geometry_config: str | Path | None = None,
        task_requirements: str | Path | dict[str, Any] | None = None,
        semantic_detector: SemanticDetector | None = None,
        semantic_config: str | Path | dict[str, Any] | None = None,
        grounding_mode: str = "geometry-only",
        pairing_strategy: str | None = None,
        save_semantic_overlays: bool = False,
        record_oracle_diagnostics: bool = True,
        run_config: dict[str, Any] | None = None,
    ) -> "ObservedStateRun":
        if run_id is None:
            timestamp = datetime.now().astimezone().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            run_id = f"{scene.scene_name}_{timestamp}"
        run_id = _safe_run_id(run_id)
        region_ids = tuple(scene.get_region_observation_states().keys())
        return cls(
            Path(runs_root) / run_id,
            scene_name=scene.scene_name,
            region_ids=region_ids,
            initial_region_id=getattr(
                scene, "initial_observation_region", "countertop"
            ),
            voxel_size=voxel_size,
            geometry_config=geometry_config,
            task_requirements=task_requirements,
            semantic_detector=semantic_detector,
            semantic_config=semantic_config,
            grounding_mode=grounding_mode,
            pairing_strategy=pairing_strategy,
            save_semantic_overlays=save_semantic_overlays,
            record_oracle_diagnostics=record_oracle_diagnostics,
            run_config=run_config,
        )

    def observe_scene(
        self,
        scene,
        *,
        stage_label: str,
        region_opened: str | None = None,
        width: int = 640,
        height: int = 480,
    ) -> tuple[PointCloudRun, Path]:
        """Capture fresh region evidence and update every persistent output."""
        stage = self.next_stage
        safe_label = _safe_run_id(stage_label)
        stage_dir = self.stages_dir / f"{stage:03d}_{safe_label}"
        if stage_dir.exists():
            raise RuntimeError(f"Stage output already exists: {stage_dir}")
        stage_dir.mkdir(parents=True)
        inspection_region = region_opened or "INITIAL"
        cloud_run = GeometryChecker(
            scene,
            width=width,
            height=height,
            voxel_size=self.voxel_size,
        ).run_region_inspection(
            inspection_region,
            stage_output_dir=stage_dir,
        )
        stage_dir = self.update_from_inspection_run(
            scene,
            cloud_run,
            stage_label=stage_label,
            region_opened=region_opened,
        )
        return cloud_run, stage_dir

    def _oracle_source_region(self, scene, instance_id: str) -> str:
        """Return simulator provenance for evaluation diagnostics only.

        This value must never be used as planner-visible location evidence.
        Runtime location comes from the region-gated inspection that accepted
        the object's current evidence.
        """
        if hasattr(scene, "get_instance_source_region"):
            source = scene.get_instance_source_region(instance_id)
            return source if source is not None else self.initial_region_id
        sources = getattr(scene, "instance_source_regions", {})
        return sources.get(instance_id, self.initial_region_id)

    def _region_states(self, scene) -> dict[str, dict[str, Any]]:
        if hasattr(scene, "get_region_observation_states"):
            return scene.get_region_observation_states()
        provided = getattr(scene, "region_states", {})
        return {
            region_id: {
                "region_id": region_id,
                "open": bool(provided.get(region_id, {}).get("open", False)),
                "inspected": bool(
                    provided.get(region_id, {}).get("inspected", False)
                ),
            }
            for region_id in self.region_ids
        }

    def _new_object_id(self) -> str:
        """Allocate an identifier that contains no semantic category."""
        return f"object_{len(self.registry['objects']) + 1:04d}"

    def _association_token(self, instance_id: str) -> str:
        """Hash the simulator identifier so persisted state exposes no label."""
        payload = f"{self.scene_name}:{instance_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    def _load_cumulative_visualization(
        self, record: dict[str, Any] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not record:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )
        relative_path = record.get(
            "cumulative_visualization_cloud_path",
            record.get("cumulative_cloud_path"),
        )
        if relative_path is None:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )
        path = self.run_dir / relative_path
        if not path.exists():
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )
        return read_ply(path)

    # Compatibility for visualization helpers; never use this in extraction.
    _load_cumulative = _load_cumulative_visualization

    def update_from_point_cloud_run(
        self,
        scene,
        cloud_run: PointCloudRun,
        *,
        stage_label: str,
        region_opened: str | None = None,
    ) -> Path:
        """Compatibility entry point with a strict stage-evidence guard."""
        return self.update_from_inspection_run(
            scene,
            cloud_run,
            stage_label=stage_label,
            region_opened=region_opened,
        )

    def update_from_inspection_run(
        self,
        scene,
        cloud_run: PointCloudRun,
        *,
        stage_label: str,
        region_opened: str | None = None,
    ) -> Path:
        """Cache valid stage measurements and separately grow visualization."""
        if cloud_run.inspection is None:
            raise TypeError(
                "Observed-state property updates require a region-local "
                "inspection run; scene-wide and combined clouds are rejected"
            )
        inspection = cloud_run.inspection
        expected_region = region_opened or "INITIAL"
        if inspection.region_id != expected_region:
            raise ValueError(
                f"Inspection region {inspection.region_id} does not match "
                f"stage source {expected_region}"
            )
        stage = self.next_stage
        safe_label = _safe_run_id(stage_label)
        stage_dir = self.stages_dir / f"{stage:03d}_{safe_label}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        events: list[dict[str, Any]] = []
        if region_opened is not None:
            events.append(
                {
                    "stage": stage,
                    "event": "REGION_OPENED",
                    "region_id": region_opened,
                }
            )
        events.extend(
            [
                {
                    "stage": stage,
                    "event": "SETTLE_COMPLETED",
                    "region_id": expected_region,
                    "settle_steps": inspection.metadata["settle_steps"],
                },
                {
                    "stage": stage,
                    "event": "VIRTUAL_INSPECTION_RIG_POSITIONED",
                    "region_id": expected_region,
                },
                {
                    "stage": stage,
                    "event": "INSPECTION_CAPTURED",
                    "region_id": expected_region,
                    "valid_camera_count": inspection.quality[
                        "valid_camera_count"
                    ],
                },
            ]
        )

        stage_changes = {
            object_id: "previous"
            for object_id in self.registry["objects"]
        }
        stage_evidence: dict[str, MeasurementEvidence] = {}
        accepted_instance_to_object_id: dict[str, str] = {}
        accepted_metadata: list[dict[str, Any]] = []
        rejected_metadata: list[dict[str, Any]] = []
        for instance_id, evidence in inspection.evidence_clouds.items():
            instance_token = self._association_token(instance_id)
            object_id = self.registry["instance_index"].get(instance_token)
            existing = (
                self.registry["objects"].get(object_id)
                if object_id is not None
                else None
            )
            is_new = existing is None
            if is_new:
                object_id = self._new_object_id()
                self.registry["instance_index"][instance_token] = object_id
                events.append(
                    {
                        "stage": stage,
                        "event": "OBJECT_DISCOVERED",
                        "object_id": object_id,
                        "instance_token": instance_token,
                    }
                )
            relative_evidence_path = (
                stage_dir.relative_to(self.run_dir)
                / "evidence"
                / object_id
                / "fused.ply"
            )
            evidenced = evidence.with_provenance(
                source_stage=stage,
                measurement_cloud_path=relative_evidence_path.as_posix(),
            )
            stage_evidence[object_id] = evidenced
            accepted_instance_to_object_id[instance_id] = object_id
            _atomic_ply(
                self.run_dir / relative_evidence_path,
                evidenced.measurement_points,
                evidenced.measurement_colors,
            )
            _atomic_json(
                self.run_dir
                / relative_evidence_path.parent
                / "quality.json",
                evidenced.measurement_quality,
            )
            stage_measurement = extract_object_properties(
                evidenced,
                config=self.config,
            )
            _atomic_json(
                self.run_dir
                / relative_evidence_path.parent
                / "properties.json",
                stage_measurement,
            )

            previous_points, previous_colors = (
                self._load_cumulative_visualization(existing)
            )
            merged_points = np.concatenate(
                (previous_points, evidenced.measurement_points)
            )
            merged_colors = np.concatenate(
                (previous_colors, evidenced.measurement_colors)
            )
            merged_points, merged_colors = voxel_downsample(
                merged_points, merged_colors, self.voxel_size
            )
            relative_visualization_path = (
                Path("objects")
                / object_id
                / "cumulative_visualization.ply"
            )
            _atomic_ply(
                self.run_dir / relative_visualization_path,
                merged_points,
                merged_colors,
            )
            # Preserve the original output name as a marked visualization-only
            # compatibility artifact. Neither path enters the extractor.
            compatibility_cumulative_path = (
                Path("objects") / object_id / "cumulative.ply"
            )
            _atomic_ply(
                self.run_dir / compatibility_cumulative_path,
                merged_points,
                merged_colors,
            )
            observed_source_region = (
                self.initial_region_id
                if expected_region == "INITIAL"
                else expected_region
            )
            source_region = (
                observed_source_region
                if is_new
                else existing["source_region"]
            )
            quality_is_valid = bool(
                evidenced.measurement_quality.get(
                    "quality_is_valid", False
                )
            )
            measurement_keys = (
                "centroid_world_m",
                "dimensions_m",
                "principal_axis_world",
                "property_status",
                "point_count",
                "contributing_camera_count",
                "geometric_properties",
                "geometric_predicates",
                "measurement_quality",
                "measurement_cloud_path",
            )
            if existing is not None and not quality_is_valid:
                measured = {
                    key: deepcopy(existing[key])
                    for key in measurement_keys
                    if key in existing
                }
                last_property_update_stage = existing.get(
                    "last_property_update_stage"
                )
                last_property_source_region = existing.get(
                    "last_property_source_region"
                )
            else:
                measured = stage_measurement
                last_property_update_stage = (
                    stage if quality_is_valid else None
                )
                last_property_source_region = (
                    expected_region if quality_is_valid else None
                )
            record = {
                "object_id": object_id,
                "instance_token": instance_token,
                "inference_basis": (
                    self.grounding_mode.upper().replace("-", "_")
                ),
                "source_region": source_region,
                "source_region_basis": "REGION_GATED_OBSERVATION",
                "first_seen_stage": stage if is_new else existing["first_seen_stage"],
                "last_seen_stage": stage,
                "observation_count": (
                    1 if is_new else int(existing["observation_count"]) + 1
                ),
                "last_evidence_stage": stage,
                "last_evidence_source_region": expected_region,
                "last_property_update_stage": last_property_update_stage,
                "last_property_source_region": last_property_source_region,
                "cumulative_visualization_cloud_path": (
                    relative_visualization_path.as_posix()
                ),
                "cumulative_cloud_path": (
                    compatibility_cumulative_path.as_posix()
                ),
                "cumulative_cloud_purpose": (
                    CUMULATIVE_VISUALIZATION_PURPOSE
                ),
                "cumulative_visualization_point_count": len(merged_points),
                "semantics": deepcopy(
                    existing.get("semantics", {})
                    if existing is not None
                    else {}
                ),
                "functional_role_evaluations": deepcopy(
                    existing.get("functional_role_evaluations", {})
                    if existing is not None
                    else {}
                ),
                "functional_usage_evaluations": deepcopy(
                    existing.get("functional_usage_evaluations", {})
                    if existing is not None
                    else {}
                ),
                "target_assignment_evaluations": deepcopy(
                    existing.get("target_assignment_evaluations", {})
                    if existing is not None
                    else {}
                ),
                **measured,
            }
            if self.record_oracle_diagnostics:
                record["oracle_source_region"] = (
                    self._oracle_source_region(scene, instance_id)
                    if is_new
                    else existing.get("oracle_source_region")
                )
                record["oracle_source_region_usage"] = "EVALUATION_ONLY"
            record["geometry"] = {
                "centroid_world_m": deepcopy(
                    record.get("centroid_world_m", {})
                ),
                "dimensions_m": deepcopy(record.get("dimensions_m", {})),
                "principal_axis_world": deepcopy(
                    record.get("principal_axis_world", {})
                ),
                "properties": deepcopy(
                    record.get("geometric_properties", {})
                ),
                "predicates": deepcopy(
                    record.get("geometric_predicates", {})
                ),
                "measurement_quality": deepcopy(
                    record.get("measurement_quality", {})
                ),
                "measurement_cloud_path": record.get(
                    "measurement_cloud_path"
                ),
            }
            self.registry["objects"][object_id] = record
            _atomic_json(
                self.run_dir / "objects" / object_id / "properties.json",
                record,
            )
            stage_changes[object_id] = "new" if is_new else "updated"
            events.append(
                {
                    "stage": stage,
                    "event": "OBJECT_EVIDENCE_ACCEPTED",
                    "region_id": expected_region,
                    "object_id": object_id,
                    "point_count": len(evidenced.measurement_points),
                }
            )
            if quality_is_valid:
                events.append(
                    {
                        "stage": stage,
                        "event": "PROPERTY_UPDATED",
                        "object_id": object_id,
                        "property": "OPEN_CAVITY",
                        "value": stage_measurement.get(
                            "geometric_predicates", {}
                        )
                        .get("OPEN_CAVITY", {})
                        .get("value"),
                        "source_region": expected_region,
                        "measurement_cloud_path": (
                            relative_evidence_path.as_posix()
                        ),
                    }
                )
            else:
                events.append(
                    {
                        "stage": stage,
                        "event": "OBJECT_EVIDENCE_INADEQUATE",
                        "region_id": expected_region,
                        "object_id": object_id,
                        "reasons": evidenced.measurement_quality.get(
                            "reasons", []
                        ),
                    }
                )
            accepted_metadata.append(
                {
                    "object_id": object_id,
                    "point_count": len(evidenced.measurement_points),
                    "quality_is_valid": quality_is_valid,
                    "measurement_cloud_path": (
                        relative_evidence_path.as_posix()
                    ),
                }
            )

        for instance_id, rejected in inspection.rejected_clouds.items():
            instance_token = self._association_token(instance_id)
            object_id = self.registry["instance_index"].get(instance_token)
            reasons = (
                inspection.quality.get(
                    "_rejected_instance_reasons", {}
                ).get(instance_id, ["OUTSIDE_INSPECTION_VOLUME"])
            )
            inside_count = int(sum(rejected.pixels_by_camera.values()))
            event = {
                "stage": stage,
                "event": "OBJECT_OUTSIDE_INSPECTION_VOLUME",
                "region_id": expected_region,
                "instance_token": instance_token,
                "point_count_inside": inside_count,
                "raw_visible_point_count": len(rejected.points),
                "reasons": reasons,
            }
            metadata_record = {
                "instance_token": instance_token,
                "point_count_inside": inside_count,
                "raw_visible_point_count": len(rejected.points),
                "reasons": reasons,
            }
            if object_id is not None:
                event["object_id"] = object_id
                metadata_record["object_id"] = object_id
            events.append(event)
            rejected_metadata.append(metadata_record)

        semantic_stage_result = None
        if self.semantic_enabled:
            role_rank_by_label = {
                role: {
                    str(label).strip().lower(): int(preference["rank"])
                    for preference in role_config.get(
                        "semantic_preferences", []
                    )
                    for label in (
                        preference["canonical_label"],
                        *preference.get("detector_aliases", []),
                    )
                }
                for role, role_config in self.task_requirements.get(
                    "roles", {}
                ).items()
            }
            semantic_stage_result = run_semantic_inspection(
                inspection,
                accepted_instance_to_object_id=(
                    accepted_instance_to_object_id
                ),
                detector=self.semantic_detector,
                config=self.semantic_config,
                stage=stage,
                region_id=expected_region,
                stage_dir=stage_dir,
                save_overlays=self.save_semantic_overlays,
                role_rank_by_label=role_rank_by_label,
            )
            stage_prefix = stage_dir.relative_to(self.run_dir)
            for summary in semantic_stage_result["camera_summaries"]:
                events.append(
                    {
                        "stage": stage,
                        "event": "SEMANTIC_DETECTION_COMPLETED",
                        "camera_id": summary["camera_id"],
                        "detection_count": summary["detection_count"],
                        "detector": semantic_stage_result["detector"][
                            "name"
                        ],
                    }
                )
            for camera_associations in semantic_stage_result[
                "associations"
            ]:
                for association in camera_associations["accepted"]:
                    detection = association["detection"]
                    events.append(
                        {
                            "stage": stage,
                            "event": "SEMANTIC_EVIDENCE_ASSOCIATED",
                            "camera_id": camera_associations[
                                "camera_id"
                            ],
                            "object_id": association["object_id"],
                            "label": detection["canonical_label"],
                            "confidence": detection["confidence"],
                            "association_score": association[
                                "association_score"
                            ],
                        }
                    )
            for object_id, observation in semantic_stage_result[
                "semantic_records"
            ].items():
                observation = deepcopy(observation)
                if observation.get("semantic_record_path"):
                    observation["semantic_record_path"] = (
                        stage_prefix
                        / observation["semantic_record_path"]
                    ).as_posix()
                observation["semantic_evidence_paths"] = [
                    (stage_prefix / path).as_posix()
                    for path in observation.get(
                        "semantic_evidence_paths", []
                    )
                ]
                record = self.registry["objects"][object_id]
                namespace = deepcopy(record.get("semantics", {}))
                history = list(namespace.get("observations", []))
                history.append(observation)
                previous_validated = namespace.get("validated")
                validated, replaced = _select_validated_semantic(
                    previous_validated, observation
                )
                if replaced:
                    namespace["last_validated_stage"] = stage
                    namespace["last_validated_source_region"] = (
                        expected_region
                    )
                if validated is not None:
                    namespace["validated"] = validated
                namespace["latest_observation"] = observation
                namespace["observations"] = history
                record["semantics"] = namespace
                _atomic_json(
                    self.run_dir
                    / observation["semantic_record_path"],
                    observation,
                )
                _atomic_json(
                    self.run_dir
                    / "objects"
                    / object_id
                    / "properties.json",
                    record,
                )

        region_states = self._region_states(scene)
        graph = self._build_graph(region_states, stage_changes)
        pair_relation_names = {
            constraint["relation"]
            for constraint in self.task_requirements["constraints"][
                "pairwise"
            ]
        }
        pair_relations = [
            {
                "source_object_id": edge["source"].removeprefix(
                    "object:"
                ),
                "target_object_id": edge["target"].removeprefix(
                    "object:"
                ),
                "relation": edge["relation"],
                "status": edge["status"],
                "evidence": deepcopy(edge.get("evidence", {})),
                "applicable_role_pairs": deepcopy(
                    edge.get("applicable_role_pairs", [])
                ),
                "role_projection_eligible": bool(
                    edge.get("role_projection_eligible", False)
                ),
            }
            for edge in graph["edges"]
            if edge.get("relation") in pair_relation_names
            and edge.get("pairing_scope")
            in {
                "ALL_OBSERVED_ORDERED_OBJECT_PAIRS",
                "SEMANTIC_ROLE_SCOPED_OBJECT_PAIRS",
            }
        ]
        pairing_metadata = deepcopy(graph.get("pairing", {}))
        pair_payload = {
            "schema_version": 1,
            "task_id": self.task_requirements["task_id"],
            "stage": stage,
            "region_id": expected_region,
            "pairing_strategy": self.pairing_strategy,
            "pairing_scope": pairing_metadata.get("scope"),
            "role_binding_phase": (
                "AFTER_PAIRWISE_GEOMETRY"
                if self.pairing_strategy == "exhaustive_all_pairs"
                else "AFTER_SEMANTIC_GATING_AND_PAIRWISE_GEOMETRY"
            ),
            "unary_geometry_scope": "ALL_OBSERVED_OBJECTS",
            "observed_object_ids": sorted(
                self.registry["objects"]
            ),
            "ordered_distinct_object_pair_count": (
                len(self.registry["objects"])
                * max(0, len(self.registry["objects"]) - 1)
            ),
            "relation_names": sorted(pair_relation_names),
            "relation_evaluation_count": len(pair_relations),
            "skipped_relation_pair_count": pairing_metadata.get(
                "skipped_relation_pair_count", 0
            ),
            "elapsed_seconds": pairing_metadata.get(
                "elapsed_seconds", 0.0
            ),
            "relations": pair_relations,
        }
        _atomic_json(
            stage_dir / "pair_relation_evaluations.json",
            pair_payload,
        )
        _atomic_json(
            self.run_dir / "pair_relation_evaluations.json",
            pair_payload,
        )
        # Preserve the exhaustive-ablation artifact name for existing reports.
        if self.pairing_strategy == "exhaustive_all_pairs":
            _atomic_json(
                stage_dir / "all_observed_pair_relations.json",
                pair_payload,
            )
            _atomic_json(
                self.run_dir / "all_observed_pair_relations.json",
                pair_payload,
            )
        events.append(
            {
                "stage": stage,
                "region_id": expected_region,
                "event": (
                    "ALL_OBSERVED_OBJECT_PAIRS_EVALUATED"
                    if self.pairing_strategy == "exhaustive_all_pairs"
                    else "SEMANTIC_ROLE_SCOPED_PAIRS_EVALUATED"
                ),
                "pairing_strategy": self.pairing_strategy,
                "observed_object_count": len(
                    self.registry["objects"]
                ),
                "ordered_distinct_object_pair_count": (
                    pair_payload[
                        "ordered_distinct_object_pair_count"
                    ]
                ),
                "relation_evaluation_count": len(
                    pair_relations
                ),
                "skipped_relation_pair_count": pair_payload[
                    "skipped_relation_pair_count"
                ],
                "elapsed_seconds": pair_payload["elapsed_seconds"],
                "role_binding_phase": (
                    pair_payload["role_binding_phase"]
                ),
            }
        )
        previous_witness_status = (
            self.latest_witness.get("status")
            if self.latest_witness is not None
            else None
        )
        task_schema = self.task_requirements.get("_task_schema")
        if task_schema == "JOINT_USAGE_POLICY_GROUNDING":
            witness = evaluate_usage_policy_task_witness(
                graph,
                self.task_requirements,
                stage=stage,
                usage_policy_mode="function-aware",
                target_assignment_mode=(
                    "joint-target-specific"
                    if self.task_requirements.get(
                        "target_assignment_ablation", False
                    )
                    else None
                ),
            )
        elif task_schema == "JOINT_ROLE_GROUNDING":
            witness = evaluate_joint_task_witness(
                graph,
                self.task_requirements,
                stage=stage,
                grounding_mode=self.grounding_mode,
            )
        else:
            witness = evaluate_task_witness(
                graph,
                self.task_requirements,
                stage=stage,
            )
        matrix = (
            build_target_compatibility_matrix(witness)
            if self.task_requirements.get(
                "target_assignment_ablation", False
            )
            else None
        )
        _atomic_json(
            stage_dir / "candidate_evaluations.json",
            {
                "task_id": witness["task_id"],
                "stage": stage,
                "grounding_mode": self.grounding_mode,
                "candidate_evaluations": witness.get(
                    "candidate_evaluations", []
                ),
                "assignment_evaluations": witness.get(
                    "assignment_evaluations", []
                ),
            },
        )
        for candidate in witness.get("candidate_evaluations", []):
            decision = candidate.get("decision")
            if decision == "REJECTED_SEMANTIC":
                events.append(
                    {
                        "stage": stage,
                        "event": "CANDIDATE_REJECTED_SEMANTIC",
                        "object_id": candidate["object_id"],
                        "role": candidate["role"],
                    }
                )
                if self.task_requirements.get(
                    "target_assignment_ablation", False
                ):
                    events.append(
                        {
                            "stage": stage,
                            "region_id": expected_region,
                            "event": (
                                "TARGET_COMPATIBILITY_REJECTED_SEMANTIC"
                            ),
                            "tool_object_id": candidate["object_id"],
                            "role": candidate["role"],
                            "target_object_id": None,
                            "status": "FALSE",
                            "rejection_reason": (
                                "SEMANTIC_ROLE_GATE_FAILED_BEFORE_PAIRING"
                            ),
                            "semantic_evidence": deepcopy(
                                candidate.get("semantic")
                            ),
                        }
                    )
            elif decision == "REJECTED_GEOMETRY":
                events.append(
                    {
                        "stage": stage,
                        "event": "CANDIDATE_REJECTED_GEOMETRY",
                        "object_id": candidate["object_id"],
                        "role": candidate["role"],
                    }
                )
        # Unary candidate checks cannot expose relation failures because
        # relations are evaluated only after a distinct multi-role assignment
        # is assembled. Emit the required relation-level rejection trace from
        # those assignment evaluations and deterministically de-duplicate
        # equivalent failures.
        emitted_relation_rejections: set[
            tuple[str, str, str, str]
        ] = set()
        for assignment in witness.get("assignment_evaluations", []):
            if assignment.get("decision") != "REJECTED_GEOMETRY":
                continue
            for relation in assignment.get("relation_checks", []):
                if relation.get("status") != "FALSE":
                    continue
                key = (
                    str(relation.get("from_object")),
                    str(relation.get("from_role")),
                    str(relation.get("relation")),
                    str(relation.get("to_object")),
                )
                if key in emitted_relation_rejections:
                    continue
                emitted_relation_rejections.add(key)
                events.append(
                    {
                        "stage": stage,
                        "event": "CANDIDATE_REJECTED_GEOMETRY",
                        "object_id": relation["from_object"],
                        "role": relation["from_role"],
                        "failed_relation": relation["relation"],
                        "related_object_id": relation["to_object"],
                        "related_role": relation["to_role"],
                    }
                )

        for candidate in witness.get("candidate_evaluations", []):
            record = self.registry["objects"][candidate["object_id"]]
            evaluations = deepcopy(
                record.get("functional_role_evaluations", {})
            )
            evaluations[candidate["role"]] = deepcopy(candidate)
            record["functional_role_evaluations"] = evaluations
        if task_schema == "JOINT_USAGE_POLICY_GROUNDING":
            if self.task_requirements.get(
                "target_assignment_ablation", False
            ):
                function_candidate_edges: set[tuple[str, str]] = set()
                for cell in matrix["cells"]:
                    status = cell[
                        "target_specific_compatibility_status"
                    ]
                    rejection = cell["rejection_reason"]
                    event = "TARGET_COMPATIBILITY_EVALUATED"
                    if status == "TRUE":
                        event = "TARGET_COMPATIBILITY_ACCEPTED"
                    elif rejection == "SEMANTIC_REJECTION":
                        event = (
                            "TARGET_COMPATIBILITY_REJECTED_SEMANTIC"
                        )
                    elif rejection == "INSERTION_REJECTION":
                        event = (
                            "TARGET_COMPATIBILITY_REJECTED_INSERTION"
                        )
                    elif rejection == "REACH_REJECTION":
                        event = "TARGET_COMPATIBILITY_REJECTED_REACH"
                    compatibility_event = {
                            "stage": stage,
                            "region_id": expected_region,
                            "function_group": cell[
                                "function_group_id"
                            ],
                            "tool_object_id": cell["tool_object_id"],
                            "target_object_id": cell[
                                "target_object_id"
                            ],
                            "semantic_status": cell[
                                "tool_semantic_status"
                            ],
                            "elongated_object_status": cell[
                                "elongated_object_status"
                            ],
                            "insertable_in_status": cell[
                                "insertable_in_status"
                            ],
                            "insertable_in_pass_margin_m": cell[
                                "insertable_in_pass_margin_m"
                            ],
                            "reaches_bottom_status": cell[
                                "reaches_bottom_status"
                            ],
                            "reaches_bottom_pass_margin_m": cell[
                                "reaches_bottom_pass_margin_m"
                            ],
                            "status": status,
                            "rejection_reason": rejection,
                    }
                    events.append(
                        {
                            **compatibility_event,
                            "event": "TARGET_COMPATIBILITY_EVALUATED",
                        }
                    )
                    if event != "TARGET_COMPATIBILITY_EVALUATED":
                        events.append(
                            {**compatibility_event, "event": event}
                        )
                    graph["edges"].append(
                        {
                            "source": f"object:{cell['tool_object_id']}",
                            "target": f"object:{cell['target_object_id']}",
                            "relation": (
                                "COMPATIBLE_WITH_TARGET"
                                if status == "TRUE"
                                else "INCOMPATIBLE_WITH_TARGET"
                                if status == "FALSE"
                                else "UNRESOLVED_WITH_TARGET"
                            ),
                            "status": status,
                            "function_group": cell[
                                "function_group_id"
                            ],
                            "evidence": deepcopy(cell),
                        }
                    )
                    candidate_key = (
                        cell["tool_object_id"],
                        cell["function_group_id"],
                    )
                    if candidate_key not in function_candidate_edges:
                        function_candidate_edges.add(candidate_key)
                        graph["edges"].append(
                            {
                                "source": (
                                    f"object:{cell['tool_object_id']}"
                                ),
                                "target": (
                                    "function_group:"
                                    f"{cell['function_group_id']}"
                                ),
                                "relation": "CANDIDATE_FOR_FUNCTION",
                                "status": _combined_candidate_status(
                                    cell["tool_semantic_status"],
                                    cell["elongated_object_status"],
                                ),
                            }
                        )
                    for object_id in (
                        cell["tool_object_id"],
                        cell["target_object_id"],
                    ):
                        record = self.registry["objects"].get(object_id)
                        if record is None:
                            continue
                        namespace = deepcopy(
                            record.get(
                                "target_assignment_evaluations", {}
                            )
                        )
                        namespace[
                            f"{cell['function_group_id']}:"
                            f"{cell['tool_object_id']}:"
                            f"{cell['target_object_id']}"
                        ] = deepcopy(cell)
                        record["target_assignment_evaluations"] = (
                            namespace
                        )
                events.append(
                    {
                        "stage": stage,
                        "region_id": expected_region,
                        "event": (
                            "TARGET_ASSIGNMENT_COMPLETE"
                            if witness["status"] == "COMPLETE"
                            else "NO_COMPLETE_TARGET_MATCHING"
                        ),
                        "task_id": witness["task_id"],
                        "reason_codes": witness["reason_codes"],
                    }
                )
            for group in witness.get(
                "function_group_evaluations", []
            ):
                group_id = group["function_group_id"]
                involved_ids = {
                    *group.get("raw_utensil_object_ids", []),
                    *group.get("eligible_target_object_ids", []),
                    *group.get("unknown_target_object_ids", []),
                }
                for object_id in involved_ids:
                    if object_id not in self.registry["objects"]:
                        continue
                    record = self.registry["objects"][object_id]
                    usage = deepcopy(
                        record.get("functional_usage_evaluations", {})
                    )
                    usage[group_id] = {
                        "stage": stage,
                        "function": group["function"],
                        "evaluated_usage_policy_mode": (
                            group["evaluated_usage_policy_mode"]
                        ),
                        "status": group["status"],
                        "counts": deepcopy(group["counts"]),
                        "assignments": [
                            deepcopy(assignment)
                            for assignment in group.get(
                                "selected_assignments", []
                            )
                            if object_id
                            in {
                                assignment["utensil_object_id"],
                                assignment["target_object_id"],
                            }
                        ],
                    }
                    record["functional_usage_evaluations"] = usage
        for node in graph["nodes"]:
            if node.get("type") != "object":
                continue
            object_id = node["attributes"]["object_id"]
            node["attributes"]["functional_role_evaluations"] = deepcopy(
                self.registry["objects"][object_id].get(
                    "functional_role_evaluations", {}
                )
            )
            node["attributes"]["functional_usage_evaluations"] = deepcopy(
                self.registry["objects"][object_id].get(
                    "functional_usage_evaluations", {}
                )
            )
            node["attributes"]["target_assignment_evaluations"] = deepcopy(
                self.registry["objects"][object_id].get(
                    "target_assignment_evaluations", {}
                )
            )

        mode_witnesses = {}
        if task_schema == "JOINT_USAGE_POLICY_GROUNDING":
            if self.task_requirements.get(
                "target_assignment_ablation", False
            ):
                assignment_modes = (
                    "semantic-only",
                    "geometry-only",
                    "joint-target-agnostic-count",
                    "joint-target-specific",
                )
                assignment_witnesses = {
                    mode: evaluate_usage_policy_task_witness(
                        graph,
                        self.task_requirements,
                        stage=stage,
                        usage_policy_mode="function-aware",
                        target_assignment_mode=mode,
                    )
                    for mode in assignment_modes
                }
                matrix = build_target_compatibility_matrix(
                    assignment_witnesses["joint-target-specific"]
                )
                # Pairing-strategy diagnostics reuse this exact registry and
                # cached stage-local measurements. They rebuild only the
                # binary relation layer; RGB-D, semantics, and unary geometry
                # are never rerun. The production strategy still controls the
                # persisted graph and inspection stopping decision.
                alternate_strategy = (
                    "exhaustive_all_pairs"
                    if self.pairing_strategy
                    == "semantic_role_scoped"
                    else "semantic_role_scoped"
                )
                production_strategy = self.pairing_strategy
                try:
                    self.pairing_strategy = alternate_strategy
                    alternate_graph = self._build_graph(
                        region_states, stage_changes
                    )
                finally:
                    self.pairing_strategy = production_strategy
                alternate_witness = evaluate_usage_policy_task_witness(
                    alternate_graph,
                    self.task_requirements,
                    stage=stage,
                    usage_policy_mode="function-aware",
                    target_assignment_mode="joint-target-specific",
                )
                alternate_matrix = build_target_compatibility_matrix(
                    alternate_witness
                )

                def _production_cells(payload):
                    return {
                        (
                            cell["function_group_id"],
                            cell["tool_object_id"],
                            cell["target_object_id"],
                        ): cell["target_specific_compatibility_status"]
                        for cell in payload.get("cells", [])
                        if cell.get("role_relevant_projection", True)
                        and cell.get("tool_semantic_status") == "TRUE"
                        and cell.get("target_semantic_status") == "TRUE"
                        and cell.get("elongated_object_status") == "TRUE"
                        and cell.get("open_cavity_status") == "TRUE"
                    }

                strategy_graphs = {
                    production_strategy: graph,
                    alternate_strategy: alternate_graph,
                }
                strategy_witnesses = {
                    production_strategy: witness,
                    alternate_strategy: alternate_witness,
                }
                strategy_matrices = {
                    production_strategy: matrix,
                    alternate_strategy: alternate_matrix,
                }
                scoped_cells = _production_cells(
                    strategy_matrices["semantic_role_scoped"]
                )
                exhaustive_cells = _production_cells(
                    strategy_matrices["exhaustive_all_pairs"]
                )
                strategy_comparison = {
                    "schema_version": 1,
                    "stage": stage,
                    "region_id": expected_region,
                    "same_saved_observation_evidence": True,
                    "perception_rerun_per_strategy": False,
                    "unary_geometry_rerun_per_strategy": False,
                    "strategies": {
                        strategy: {
                            "status": strategy_witnesses[strategy][
                                "status"
                            ],
                            "selected_witness": strategy_witnesses[
                                strategy
                            ]["selected_witness"],
                            "operation_assignments": strategy_witnesses[
                                strategy
                            ]["operation_assignments"],
                            "compatibility_matrix": strategy_matrices[
                                strategy
                            ],
                            **deepcopy(
                                strategy_graphs[strategy].get(
                                    "pairing", {}
                                )
                            ),
                        }
                        for strategy in (
                            "exhaustive_all_pairs",
                            "semantic_role_scoped",
                        )
                    },
                    "required_production_edge_matrices_identical": (
                        scoped_cells == exhaustive_cells
                    ),
                    "final_status_identical": (
                        strategy_witnesses["semantic_role_scoped"][
                            "status"
                        ]
                        == strategy_witnesses["exhaustive_all_pairs"][
                            "status"
                        ]
                    ),
                    "selected_assignments_identical": (
                        sorted(
                            (
                                item["function_group_id"],
                                item["utensil_object_id"],
                                item["target_object_id"],
                            )
                            for item in strategy_witnesses[
                                "semantic_role_scoped"
                            ]["operation_assignments"]
                        )
                        == sorted(
                            (
                                item["function_group_id"],
                                item["utensil_object_id"],
                                item["target_object_id"],
                            )
                            for item in strategy_witnesses[
                                "exhaustive_all_pairs"
                            ]["operation_assignments"]
                        )
                    ),
                }
                _atomic_json(
                    stage_dir / "pairing_strategy_comparison.json",
                    strategy_comparison,
                )
                _atomic_json(
                    self.run_dir / "pairing_strategy_comparison.json",
                    strategy_comparison,
                )
                evidence_manifest = {
                    "measurement_cloud_paths": sorted(
                        evidence.measurement_cloud_path
                        for evidence in stage_evidence.values()
                        if evidence.measurement_cloud_path
                    ),
                    "semantic_record_paths": sorted(
                        record.get("semantics", {})
                        .get("latest_observation", {})
                        .get("semantic_record_path")
                        for record in self.registry["objects"].values()
                        if record.get("semantics", {})
                        .get("latest_observation", {})
                        .get("semantic_record_path")
                    ),
                }
                assignment_comparison = {
                    "stage": stage,
                    "region_id": expected_region,
                    "same_observation_evidence": True,
                    "perception_rerun_per_mode": False,
                    "production_mode": "joint-target-specific",
                    **evidence_manifest,
                    "modes": assignment_witnesses,
                }
                _atomic_json(
                    stage_dir / "assignment_mode_comparison.json",
                    assignment_comparison,
                )
                _atomic_json(
                    stage_dir / "assignment_evaluations.json",
                    {
                        "task_id": witness["task_id"],
                        "stage": stage,
                        "production_mode": "joint-target-specific",
                        "modes": assignment_witnesses,
                    },
                )
                _atomic_json(
                    stage_dir / "compatibility_matrix.json", matrix
                )
                _atomic_compatibility_csv(
                    stage_dir / "compatibility_matrix.csv", matrix
                )
                _atomic_json(
                    self.run_dir / "compatibility_matrix.json", matrix
                )
                _atomic_compatibility_csv(
                    self.run_dir / "compatibility_matrix.csv", matrix
                )
                _atomic_json(
                    self.run_dir / "assignment_evaluations.json",
                    {
                        "task_id": witness["task_id"],
                        "stage": stage,
                        "production_mode": "joint-target-specific",
                        "modes": assignment_witnesses,
                    },
                )
                assignment_summary_path = (
                    self.run_dir / "assignment_ablation_summary.json"
                )
                assignment_summary = (
                    json.loads(assignment_summary_path.read_text())
                    if assignment_summary_path.exists()
                    else {
                        "task_id": witness["task_id"],
                        "production_mode": "joint-target-specific",
                        "diagnostic_modes": list(assignment_modes[:-1]),
                        "shared_observation_evidence": True,
                        "stages": [],
                    }
                )
                assignment_summary["stages"] = [
                    item
                    for item in assignment_summary["stages"]
                    if int(item["stage"]) != stage
                ]
                assignment_summary["stages"].append(
                    {
                        "stage": stage,
                        "region_id": expected_region,
                        "evidence": evidence_manifest,
                        "compatibility_matrix_path": (
                            stage_dir.relative_to(self.run_dir)
                            / "compatibility_matrix.json"
                        ).as_posix(),
                        "modes": {
                            mode: {
                                "status": result["status"],
                                "selected_witness": result[
                                    "selected_witness"
                                ],
                                "operation_assignments": result[
                                    "operation_assignments"
                                ],
                                "reason_codes": result["reason_codes"],
                            }
                            for mode, result in assignment_witnesses.items()
                        },
                    }
                )
                assignment_summary["stages"].sort(
                    key=lambda item: item["stage"]
                )
                _atomic_json(
                    assignment_summary_path, assignment_summary
                )
            policy_witnesses = {
                mode: evaluate_usage_policy_task_witness(
                    graph,
                    self.task_requirements,
                    stage=stage,
                    usage_policy_mode=mode,
                )
                for mode in (
                    "function-aware",
                    "always-reusable",
                    "always-distinct",
                )
            }
            comparison = {
                "stage": stage,
                "same_observation_evidence": True,
                "production_mode": "function-aware",
                "measurement_cloud_paths": sorted(
                    evidence.measurement_cloud_path
                    for evidence in stage_evidence.values()
                    if evidence.measurement_cloud_path
                ),
                "semantic_record_paths": sorted(
                    record.get("semantics", {})
                    .get("latest_observation", {})
                    .get("semantic_record_path")
                    for record in self.registry["objects"].values()
                    if record.get("semantics", {})
                    .get("latest_observation", {})
                    .get("semantic_record_path")
                ),
                "modes": policy_witnesses,
            }
            _atomic_json(
                stage_dir / "policy_mode_comparison.json", comparison
            )
            _atomic_json(
                stage_dir / "usage_policy_evaluations.json",
                {
                    "task_id": witness["task_id"],
                    "stage": stage,
                    "production_mode": "function-aware",
                    "function_groups": witness[
                        "function_group_evaluations"
                    ],
                },
            )
            _atomic_json(
                stage_dir / "operation_assignments.json",
                {
                    "task_id": witness["task_id"],
                    "stage": stage,
                    "usage_policy_mode": "function-aware",
                    "assignments": witness["operation_assignments"],
                },
            )
            _atomic_json(
                stage_dir / "function_assignments.json",
                {
                    "task_id": witness["task_id"],
                    "stage": stage,
                    "usage_policy_mode": "function-aware",
                    "assignments": witness["operation_assignments"],
                    "function_groups": witness[
                        "function_group_evaluations"
                    ],
                },
            )
            _atomic_json(
                self.run_dir / "usage_policy_evaluations.json",
                {
                    "task_id": witness["task_id"],
                    "stage": stage,
                    "production_mode": "function-aware",
                    "function_groups": witness[
                        "function_group_evaluations"
                    ],
                },
            )
            _atomic_json(
                self.run_dir / "operation_assignments.json",
                {
                    "task_id": witness["task_id"],
                    "stage": stage,
                    "usage_policy_mode": "function-aware",
                    "assignments": witness["operation_assignments"],
                },
            )
            _atomic_json(
                self.run_dir / "function_assignments.json",
                {
                    "task_id": witness["task_id"],
                    "stage": stage,
                    "usage_policy_mode": "function-aware",
                    "assignments": witness["operation_assignments"],
                    "function_groups": witness[
                        "function_group_evaluations"
                    ],
                },
            )
            policy_path = self.run_dir / "policy_ablation_summary.json"
            policy_summary = (
                json.loads(policy_path.read_text())
                if policy_path.exists()
                else {
                    "task_id": witness["task_id"],
                    "production_mode": "function-aware",
                    "diagnostic_modes": [
                        "always-reusable",
                        "always-distinct",
                    ],
                    "shared_observation_evidence": True,
                    "stages": [],
                }
            )
            policy_summary["stages"] = [
                item
                for item in policy_summary["stages"]
                if int(item["stage"]) != stage
            ]
            policy_summary["stages"].append(
                {
                    "stage": stage,
                    "region_id": expected_region,
                    "modes": {
                        mode: {
                            "status": result["status"],
                            "distinct_physical_tool_count": result[
                                "distinct_physical_tool_count"
                            ],
                            "policy_required_distinct_physical_tool_count": (
                                result[
                                    "policy_required_distinct_physical_tool_count"
                                ]
                            ),
                            "satisfied_target_slot_count": result[
                                "satisfied_target_slot_count"
                            ],
                            "required_target_slot_count": result[
                                "required_target_slot_count"
                            ],
                            "operation_assignments": result[
                                "operation_assignments"
                            ],
                            "reason_codes": result["reason_codes"],
                        }
                        for mode, result in policy_witnesses.items()
                    },
                }
            )
            policy_summary["stages"].sort(
                key=lambda item: item["stage"]
            )
            _atomic_json(policy_path, policy_summary)
            diagnostic_payload = {
                "schema_version": 1,
                "task_id": witness["task_id"],
                "stage": stage,
                "region_id": expected_region,
                "same_observation_evidence": True,
                "production_status": witness["status"],
                "usage_policy_modes": {
                    mode: {
                        "status": result["status"],
                        "distinct_physical_tool_count": result[
                            "distinct_physical_tool_count"
                        ],
                    }
                    for mode, result in policy_witnesses.items()
                },
            }
            strategy_path = (
                stage_dir / "pairing_strategy_comparison.json"
            )
            if strategy_path.exists():
                diagnostic_payload["pairing_strategies"] = json.loads(
                    strategy_path.read_text(encoding="utf-8")
                )
            assignment_path = (
                stage_dir / "assignment_mode_comparison.json"
            )
            if assignment_path.exists():
                diagnostic_payload["grounding_modes"] = json.loads(
                    assignment_path.read_text(encoding="utf-8")
                )["modes"]
            _atomic_json(
                stage_dir / "diagnostic_summary.json",
                diagnostic_payload,
            )
            _atomic_json(
                self.run_dir / "diagnostic_summary.json",
                diagnostic_payload,
            )
        elif task_schema == "JOINT_ROLE_GROUNDING":
            # Production uses semantic-role-scoped binary checks to avoid
            # spending relation work on semantically irrelevant pairs.  The
            # geometry-only ablation has deliberately removed that semantic
            # gate, so evaluating it on the production graph would leave
            # those pruned pairs UNKNOWN. Rebuild only the binary relation
            # layer from the same cached stage-local measurements; RGB-D,
            # semantic inference, and unary extraction are not repeated.
            geometry_ablation_graph = graph
            if self.pairing_strategy != "exhaustive_all_pairs":
                production_strategy = self.pairing_strategy
                try:
                    self.pairing_strategy = "exhaustive_all_pairs"
                    geometry_ablation_graph = self._build_graph(
                        region_states, stage_changes
                    )
                finally:
                    self.pairing_strategy = production_strategy
            mode_graphs = {
                "joint": graph,
                "geometry-only": geometry_ablation_graph,
                "semantic-only": graph,
            }
            mode_witnesses = {
                mode: evaluate_joint_task_witness(
                    mode_graphs[mode],
                    self.task_requirements,
                    stage=stage,
                    grounding_mode=mode,
                )
                for mode in ("joint", "geometry-only", "semantic-only")
            }
            _atomic_json(
                stage_dir / "grounding_mode_comparison.json",
                {
                    "stage": stage,
                    "same_observation_evidence": True,
                    "perception_rerun_per_mode": False,
                    "unary_geometry_rerun_per_mode": False,
                    "binary_pairing_strategy_by_mode": {
                        mode: mode_graphs[mode]
                        .get("pairing", {})
                        .get("strategy")
                        for mode in mode_graphs
                    },
                    "measurement_cloud_paths": sorted(
                        evidence.measurement_cloud_path
                        for evidence in stage_evidence.values()
                        if evidence.measurement_cloud_path
                    ),
                    "semantic_record_paths": sorted(
                        record.get("semantics", {})
                        .get("latest_observation", {})
                        .get("semantic_record_path")
                        for record in self.registry["objects"].values()
                        if record.get("semantics", {})
                        .get("latest_observation", {})
                        .get("semantic_record_path")
                    ),
                    "modes": mode_witnesses,
                },
            )
            ablation_path = self.run_dir / "ablation_summary.json"
            ablation_summary = (
                json.loads(ablation_path.read_text())
                if ablation_path.exists()
                else {
                    "task_id": witness["task_id"],
                    "diagnostic_only": True,
                    "shared_observation_evidence": True,
                    "stages": [],
                }
            )
            ablation_summary["stages"] = [
                record
                for record in ablation_summary["stages"]
                if int(record["stage"]) != stage
            ]
            ablation_summary["stages"].append(
                {
                    "stage": stage,
                    "region_id": expected_region,
                    "modes": {
                        mode: {
                            "status": result["status"],
                            "selected_witness": result[
                                "selected_witness"
                            ],
                            "selected_candidate_edges": result[
                                "selected_candidate_edges"
                            ],
                            "reason_codes": result["reason_codes"],
                        }
                        for mode, result in mode_witnesses.items()
                    },
                }
            )
            ablation_summary["stages"].sort(
                key=lambda record: record["stage"]
            )
            _atomic_json(ablation_path, ablation_summary)

        if task_schema == "JOINT_USAGE_POLICY_GROUNDING":
            for group in witness.get(
                "function_group_evaluations", []
            ):
                events.append(
                    {
                        "stage": stage,
                        "region_id": expected_region,
                        "event": "FUNCTION_GROUP_EVALUATED",
                        "function_group": group[
                            "function_group_id"
                        ],
                        "function": group["function"],
                        "policy_mode": "function-aware",
                        "status": group["status"],
                        "counts": group["counts"],
                    }
                )
                events.append(
                    {
                        "stage": stage,
                        "region_id": expected_region,
                        "event": "USAGE_POLICY_EVALUATED",
                        "function_group": group[
                            "function_group_id"
                        ],
                        "policy_mode": "function-aware",
                        "declared_usage_policy": group[
                            "declared_usage_policy"
                        ],
                        "status": group["status"],
                        "counts": group["counts"],
                    }
                )
                events.append(
                    {
                        "stage": stage,
                        "region_id": expected_region,
                        "event": (
                            "FUNCTION_GROUP_COMPLETE"
                            if group["status"] == "COMPLETE"
                            else "FUNCTION_GROUP_INCOMPLETE"
                        ),
                        "function_group": group[
                            "function_group_id"
                        ],
                        "policy_mode": "function-aware",
                        "counts": group["counts"],
                        "reason": group["reason"],
                    }
                )
                if (
                    self.task_requirements.get(
                        "target_assignment_ablation", False
                    )
                    and group["status"] != "COMPLETE"
                ):
                    usage_mode = group["declared_usage_policy"].get(
                        "mode"
                    )
                    if usage_mode == "sequential_reuse_allowed":
                        diagnostic_event = (
                            "NO_ALL_TARGET_REUSABLE_CANDIDATE"
                        )
                    elif usage_mode == "dedicated_per_target":
                        diagnostic_event = (
                            "NO_COMPLETE_DEDICATED_MATCHING"
                        )
                    else:
                        diagnostic_event = None
                    if diagnostic_event is not None:
                        events.append(
                            {
                                "stage": stage,
                                "region_id": expected_region,
                                "event": diagnostic_event,
                                "function_group": group[
                                    "function_group_id"
                                ],
                                "policy_mode": usage_mode,
                                "counts": group["counts"],
                                "reason": group["reason"],
                            }
                        )
                count_event = (
                    "COUNT_REQUIREMENT_SATISFIED"
                    if group["counts"]["satisfied_target_slots"]
                    == group["counts"]["required_target_slots"]
                    else "COUNT_REQUIREMENT_UNSATISFIED"
                )
                events.append(
                    {
                        "stage": stage,
                        "region_id": expected_region,
                        "event": count_event,
                        "function_group": group[
                            "function_group_id"
                        ],
                        "policy_mode": "function-aware",
                        "counts": group["counts"],
                    }
                )
                if (
                    self.task_requirements.get(
                        "target_assignment_ablation", False
                    )
                    and group["status"] == "COMPLETE"
                ):
                    policy = group["declared_usage_policy"]["mode"]
                    events.append(
                        {
                            "stage": stage,
                            "region_id": expected_region,
                            "event": (
                                "REUSABLE_MULTI_TARGET_ASSIGNMENT_CREATED"
                                if policy == "sequential_reuse_allowed"
                                else "DEDICATED_TARGET_MATCHING_CREATED"
                            ),
                            "function_group": group[
                                "function_group_id"
                            ],
                            "assignments": deepcopy(
                                group.get("selected_assignments", [])
                            ),
                        }
                    )
                    for tool_id in sorted(
                        {
                            item["utensil_object_id"]
                            for item in group.get(
                                "selected_assignments", []
                            )
                        }
                    ):
                        graph["edges"].append(
                            {
                                "source": f"object:{tool_id}",
                                "target": (
                                    "function_group:"
                                    f"{group['function_group_id']}"
                                ),
                                "relation": "SATISFIES_FUNCTION_GROUP",
                                "status": "TRUE",
                            }
                        )
                for assignment in group.get(
                    "selected_assignments", []
                ):
                    events.append(
                        {
                            "stage": stage,
                            "region_id": expected_region,
                            "event": "TARGET_ASSIGNMENT_CREATED",
                            "function_group": group[
                                "function_group_id"
                            ],
                            "policy_mode": "function-aware",
                            "utensil_object_id": assignment[
                                "utensil_object_id"
                            ],
                            "target_object_id": assignment[
                                "target_object_id"
                            ],
                            "reused_assignment": assignment[
                                "reused_assignment"
                            ],
                            "dedicated_assignment": assignment[
                                "dedicated_assignment"
                            ],
                        }
                    )
                    if assignment["reused_assignment"]:
                        events.append(
                            {
                                "stage": stage,
                                "region_id": expected_region,
                                "event": "OBJECT_REUSED_FOR_TARGET",
                                "function_group": group[
                                    "function_group_id"
                                ],
                                "policy_mode": "function-aware",
                                "object_id": assignment[
                                    "utensil_object_id"
                                ],
                                "target_object_id": assignment[
                                    "target_object_id"
                                ],
                            }
                        )
                    if assignment["dedicated_assignment"]:
                        events.append(
                            {
                                "stage": stage,
                                "region_id": expected_region,
                                "event": "DEDICATED_ASSIGNMENT_CREATED",
                                "function_group": group[
                                    "function_group_id"
                                ],
                                "policy_mode": "function-aware",
                                "object_id": assignment[
                                    "utensil_object_id"
                                ],
                                "target_object_id": assignment[
                                    "target_object_id"
                                ],
                            }
                        )
                for edge in group.get(
                    "candidate_target_evaluations", []
                ):
                    if edge["status"] == "TRUE":
                        continue
                    events.append(
                        {
                            "stage": stage,
                            "region_id": expected_region,
                            "event": "CANDIDATE_EXCLUDED_FROM_COUNT",
                            "function_group": group[
                                "function_group_id"
                            ],
                            "policy_mode": "function-aware",
                            "object_id": edge[
                                "utensil_object_id"
                            ],
                            "target_object_id": edge[
                                "target_object_id"
                            ],
                            "status": edge["status"],
                            "reason": edge["reason"],
                        }
                    )
            for assignment in witness.get(
                "operation_assignments", []
            ):
                graph["edges"].append(
                    {
                        "source": (
                            f"object:{assignment['utensil_object_id']}"
                        ),
                        "target": (
                            f"object:{assignment['target_object_id']}"
                        ),
                        "relation": (
                            "DEDICATED_TO_TARGET"
                            if assignment["dedicated_assignment"]
                            else "REUSED_FOR_TARGET"
                            if assignment["reused_assignment"]
                            else "ASSIGNED_TO_TARGET"
                        ),
                        "status": "TRUE",
                        "evidence": deepcopy(assignment),
                    }
                )

        for selected in witness.get("selected_candidate_edges", []):
            graph["edges"].append(
                {
                    "source": f"object:{selected['object_id']}",
                    "target": f"role:{selected['role']}",
                    "relation": "ASSIGNED_TO_ROLE",
                    "status": "TRUE",
                    "evidence": deepcopy(selected),
                }
            )
        _atomic_json(
            self.run_dir / "candidate_evaluations.json",
            {
                "task_id": witness["task_id"],
                "stage": stage,
                "grounding_mode": self.grounding_mode,
                "candidate_evaluations": witness.get(
                    "candidate_evaluations", []
                ),
                "assignment_evaluations": witness.get(
                    "assignment_evaluations", []
                ),
            },
        )

        self.latest_witness = witness
        events.append(
            {
                "stage": stage,
                "event": "WITNESS_EVALUATED",
                "task_id": witness["task_id"],
                "status": witness["status"],
            }
        )
        if (
            witness["status"] == "COMPLETE"
            and previous_witness_status != "COMPLETE"
        ):
            events.append(
                {
                    "stage": stage,
                    "event": "COMPLETE_WITNESS_FOUND",
                    "task_id": witness["task_id"],
                }
            )
            if self.grounding_mode == "joint":
                selected_records = []
                evidence_paths = set()
                for selected in witness.get(
                    "selected_candidate_edges", []
                ):
                    object_id = selected["object_id"]
                    role = selected["role"]
                    candidate = next(
                        candidate
                        for candidate in witness[
                            "candidate_evaluations"
                        ]
                        if candidate["object_id"] == object_id
                        and candidate["role"] == role
                    )
                    object_record = self.registry["objects"][object_id]
                    measurement_path = object_record.get(
                        "measurement_cloud_path"
                    )
                    if measurement_path:
                        evidence_paths.add(measurement_path)
                    semantic = object_record.get("semantics", {}).get(
                        "validated", {}
                    )
                    if semantic.get("semantic_record_path"):
                        evidence_paths.add(
                            semantic["semantic_record_path"]
                        )
                    evidence_paths.update(
                        semantic.get("semantic_evidence_paths", [])
                    )
                    selected_records.append(
                        {
                            "role": role,
                            "object_id": object_id,
                            "semantic_rank": selected.get(
                                "semantic_rank"
                            ),
                            "semantic": candidate["semantic"],
                            "unary_geometry": candidate[
                                "unary_geometry"
                            ],
                            "measurement_cloud_path": measurement_path,
                        }
                    )
                handoff = {
                    "schema_version": 1,
                    "task_id": witness["task_id"],
                    "goal_instruction": self.task_requirements.get(
                        "goal_instruction"
                    ),
                    "manual_requirement_specification": self.task_requirements.get(
                        "specification_source"
                    ),
                    "grounding_mode": "joint",
                    "role_assignments": witness["selected_witness"],
                    "selected_roles": selected_records,
                    "relational_geometry": witness[
                        "selected_pairwise_relations"
                    ],
                    "completion_stage": stage,
                    "completion_region": expected_region,
                    "evidence_paths": sorted(
                        {
                            *evidence_paths,
                            (
                                stage_dir.relative_to(self.run_dir)
                                / "candidate_evaluations.json"
                            ).as_posix(),
                            (
                                stage_dir.relative_to(self.run_dir)
                                / "witness.json"
                            ).as_posix(),
                        }
                    ),
                    "verified": True,
                    "ready_for_tamp": True,
                    "tamp_executed": False,
                }
                if task_schema == "JOINT_USAGE_POLICY_GROUNDING":
                    handoff.update(
                        {
                            "schema_version": 2,
                            "function_groups": deepcopy(
                                witness[
                                    "function_group_evaluations"
                                ]
                            ),
                            "operation_assignments": deepcopy(
                                witness["operation_assignments"]
                            ),
                            "distinct_physical_tool_count": witness[
                                "distinct_physical_tool_count"
                            ],
                            "policy_required_distinct_physical_tool_count": (
                                witness[
                                    "policy_required_distinct_physical_tool_count"
                                ]
                            ),
                            "satisfied_target_slot_count": witness[
                                "satisfied_target_slot_count"
                            ],
                            "required_target_slot_count": witness[
                                "required_target_slot_count"
                            ],
                            "cross_group_reuse": deepcopy(
                                witness["cross_group_reuse"]
                            ),
                            "usage_policy_mode": "function-aware",
                        }
                    )
                    if self.task_requirements.get(
                        "target_assignment_ablation", False
                    ):
                        operation_assignments = witness[
                            "operation_assignments"
                        ]
                        target_ids = sorted(
                            {
                                assignment["target_object_id"]
                                for assignment in operation_assignments
                            }
                        )
                        spoon_ids = sorted(
                            {
                                assignment["utensil_object_id"]
                                for assignment in operation_assignments
                            }
                        )
                        assignments_by_group = {
                            group["function_group_id"]: [
                                deepcopy(assignment)
                                for assignment in operation_assignments
                                if assignment["function_group_id"]
                                == group["function_group_id"]
                            ]
                            for group in witness[
                                "function_group_evaluations"
                            ]
                        }
                        group_tool_sets = [
                            {
                                assignment["utensil_object_id"]
                                for assignment in assignments
                            }
                            for assignments in assignments_by_group.values()
                        ]
                        reused_across_groups = sorted(
                            set.intersection(*group_tool_sets)
                            if len(group_tool_sets) > 1
                            else set()
                        )
                        handoff.update(
                            {
                                "schema_version": 3,
                                "target_assignment_mode": (
                                    "joint-target-specific"
                                ),
                                "compatibility_matrix_reference": (
                                    "compatibility_matrix.json"
                                ),
                                "complete_target_coverage": True,
                                "target_object_ids": target_ids,
                                "selected_spoon_object_ids": spoon_ids,
                                "assignments_by_function_group": (
                                    assignments_by_group
                                ),
                                "cross_group_reused_object_ids": (
                                    reused_across_groups
                                ),
                                "total_distinct_physical_spoon_count": len(
                                    spoon_ids
                                ),
                            }
                        )
                _atomic_json(
                    self.run_dir / "verified_task_handoff.json",
                    handoff,
                )
                tool_selections = [
                    selected
                    for selected in witness.get(
                        "selected_candidate_edges", []
                    )
                    if selected["role"] == "mixing_tool"
                ]
                if (
                    tool_selections
                    and int(
                        tool_selections[0].get("semantic_rank") or 1
                    )
                    > 1
                ):
                    events.append(
                        {
                            "stage": stage,
                            "event": (
                                "ALTERNATIVE_CANDIDATE_SELECTED"
                            ),
                            **tool_selections[0],
                        }
                    )
                events.append(
                    {
                        "stage": stage,
                        "event": (
                            "VERIFIED_TASK_WITNESS_COMPLETE"
                        ),
                        "task_id": witness["task_id"],
                        "ready_for_tamp": True,
                    }
                )
        self.registry["current_stage"] = stage
        for object_id, record in self.registry["objects"].items():
            _atomic_json(
                self.run_dir
                / "objects"
                / object_id
                / "properties.json",
                record,
            )
        _atomic_json(self.registry_path, self.registry)
        _atomic_json(self.graph_path, graph)
        _atomic_json(self.latest_witness_path, witness)
        _atomic_json(stage_dir / "properties.json", self.registry)
        _atomic_json(stage_dir / "graph.json", graph)
        _atomic_json(stage_dir / "witness.json", witness)

        all_points, all_colors = self._all_cumulative_clouds()
        _atomic_ply(stage_dir / "combined_cloud.ply", all_points, all_colors)
        region_points = [
            evidence.measurement_points
            for evidence in stage_evidence.values()
            if len(evidence.measurement_points)
        ]
        region_colors = [
            evidence.measurement_colors
            for evidence in stage_evidence.values()
            if len(evidence.measurement_points)
        ]
        _atomic_ply(
            stage_dir / "region_combined_cloud.ply",
            (
                np.concatenate(region_points)
                if region_points
                else np.empty((0, 3), dtype=np.float32)
            ),
            (
                np.concatenate(region_colors)
                if region_colors
                else np.empty((0, 3), dtype=np.uint8)
            ),
        )
        inspection_metadata = deepcopy(inspection.metadata)
        inspection_metadata.update(
            {
                "stage": stage,
                "objects_accepted_inside_region": accepted_metadata,
                "objects_rejected_as_outside_region": rejected_metadata,
            }
        )
        inspection_quality = {
            key: deepcopy(value)
            for key, value in inspection.quality.items()
            if not key.startswith("_")
        }
        inspection_quality["accepted_objects"] = accepted_metadata
        inspection_quality["rejected_objects"] = rejected_metadata
        _atomic_json(
            stage_dir / "inspection_metadata.json",
            inspection_metadata,
        )
        _atomic_json(
            stage_dir / "inspection_quality.json",
            inspection_quality,
        )
        self._append_events(events)
        self._render_stage(
            stage_dir,
            graph,
            witness,
            stage_changes,
            inspection=inspection,
            stage_evidence=stage_evidence,
        )
        self._update_growth_gif()
        self.next_stage += 1
        return stage_dir

    def append_event(self, event: dict[str, Any]) -> None:
        """Append one run-level event without modifying prior history."""
        payload = dict(event)
        payload.setdefault(
            "stage", int(self.registry.get("current_stage", -1))
        )
        self._append_events([payload])

    def mark_inspection_exhausted(
        self, *, sequence: list[str], final_witness_status: str
    ) -> None:
        """Persist a terminal run outcome without falsifying witness truth.

        ``latest_witness.status`` remains the evidence-level task result
        (normally INCOMPLETE or INDETERMINATE).  EXHAUSTED describes control
        flow: every configured region was inspected without finding a complete
        witness.
        """
        stage = int(self.registry.get("current_stage", -1))
        outcome = {
            "schema_version": 1,
            "task_id": self.task_requirements["task_id"],
            "terminal_status": "EXHAUSTED",
            "final_witness_status": final_witness_status,
            "completion_stage": None,
            "completion_region": None,
            "final_stage": stage,
            "inspection_sequence": list(sequence),
            "verified": False,
            "ready_for_tamp": False,
            "tamp_executed": False,
        }
        _atomic_json(self.run_dir / "run_outcome.json", outcome)
        if self.latest_witness is not None:
            self.latest_witness = {
                **self.latest_witness,
                "terminal_status": "EXHAUSTED",
                "inspection_sequence_exhausted": True,
            }
            _atomic_json(self.latest_witness_path, self.latest_witness)
            stage_dirs = sorted(self.stages_dir.glob(f"{stage:03d}_*"))
            if stage_dirs:
                _atomic_json(
                    stage_dirs[-1] / "witness.json",
                    self.latest_witness,
                )

    def _append_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        with self.events_path.open("a", encoding="utf-8") as output:
            for event in events:
                output.write(json.dumps(event, sort_keys=True) + "\n")

    def _all_cumulative_clouds(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_points, all_colors = [], []
        for record in self.registry["objects"].values():
            points, colors = self._load_cumulative(record)
            if len(points):
                all_points.append(points)
                all_colors.append(colors)
        return (
            np.concatenate(all_points)
            if all_points
            else np.empty((0, 3), dtype=np.float32),
            np.concatenate(all_colors)
            if all_colors
            else np.empty((0, 3), dtype=np.uint8),
        )

    def _build_graph(
        self,
        region_states: dict[str, dict[str, Any]],
        stage_changes: dict[str, str],
    ) -> dict[str, Any]:
        """Build independent semantic, geometric, and relational evidence."""
        nodes, edges = [], []
        all_regions = {
            self.initial_region_id: {
                "region_id": self.initial_region_id,
                "open": True,
                "inspected": True,
            },
            **region_states,
        }
        for region_id, state in all_regions.items():
            contents = sorted(
                object_id
                for object_id, record in self.registry["objects"].items()
                if record["source_region"] == region_id
            )
            inspected = bool(state.get("inspected", False))
            nodes.append(
                {
                    "id": f"region:{region_id}",
                    "type": "region",
                    "attributes": {
                        "region_id": region_id,
                        "open": bool(state.get("open", False)),
                        "inspected": inspected,
                        "contents": contents if inspected else "UNKNOWN",
                    },
                }
            )

        role_evaluations: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        semantic_evaluations: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        roles = self.task_requirements["roles"]
        for role_name in sorted(roles):
            nodes.append(
                {
                    "id": f"role:{role_name}",
                    "type": "functional_role",
                    "attributes": {
                        "role_id": role_name,
                        "geometric_requirements": roles[role_name][
                            "geometric_requirements"
                        ],
                        "semantic_preferences": deepcopy(
                            roles[role_name].get(
                                "semantic_preferences", []
                            )
                        ),
                    },
                }
            )
        for group_id, group in sorted(
            self.task_requirements.get("operation_groups", {}).items()
        ):
            nodes.append(
                {
                    "id": f"function_group:{group_id}",
                    "type": "function_group",
                    "attributes": {
                        "function_group_id": group_id,
                        "function": group["function"],
                        "tool_role": group["tool_role"],
                        "target_role": group["target_role"],
                        "required_target_count": group[
                            "required_target_count"
                        ],
                        "usage_policy": deepcopy(
                            group["usage_policy"]
                        ),
                    },
                }
            )

        for object_id, record in sorted(self.registry["objects"].items()):
            dimensions = _dimension_values(record)
            centroid_record = record.get("centroid_world_m", {})
            nodes.append(
                {
                    "id": f"object:{object_id}",
                    "type": "object",
                    "attributes": {
                        "object_id": object_id,
                        "inference_basis": (
                            self.grounding_mode.upper().replace("-", "_")
                        ),
                        "source_region": record.get("source_region"),
                        "first_seen_stage": record.get("first_seen_stage"),
                        "last_seen_stage": record.get("last_seen_stage"),
                        "last_property_update_stage": record.get(
                            "last_property_update_stage"
                        ),
                        "last_property_source_region": record.get(
                            "last_property_source_region"
                        ),
                        "measurement_cloud_path": record.get(
                            "measurement_cloud_path"
                        ),
                        "measurement_quality": record.get(
                            "measurement_quality", {}
                        ),
                        "centroid_world_m": centroid_record.get("value"),
                        "dimensions_m": dimensions,
                        "geometric_properties": record.get(
                            "geometric_properties", {}
                        ),
                        "geometric_predicates": record.get(
                            "geometric_predicates", {}
                        ),
                        "geometry": record.get("geometry", {}),
                        "semantics": record.get("semantics", {}),
                        "functional_role_evaluations": record.get(
                            "functional_role_evaluations", {}
                        ),
                        "functional_usage_evaluations": record.get(
                            "functional_usage_evaluations", {}
                        ),
                        "target_assignment_evaluations": record.get(
                            "target_assignment_evaluations", {}
                        ),
                        "stage_state": stage_changes.get(object_id, "previous"),
                    },
                }
            )
            edges.append(
                {
                    "source": f"object:{object_id}",
                    "target": f"region:{record['source_region']}",
                    "relation": "OBSERVED_IN",
                    "status": "OBSERVED",
                }
            )
            for role_name in sorted(roles):
                evaluation = evaluate_geometric_requirements(
                    record,
                    roles[role_name]["geometric_requirements"],
                )
                role_evaluations[(object_id, role_name)] = evaluation
                edges.append(
                    {
                        "source": f"object:{object_id}",
                        "target": f"role:{role_name}",
                        "relation": "SATISFIES_GEOMETRY",
                        "status": evaluation["status"],
                        "evidence": {
                            "inference_basis": "GEOMETRY_ONLY",
                            "checks": evaluation["checks"],
                        },
                    }
                )
                if roles[role_name].get("semantic_preferences"):
                    semantic = evaluate_semantic_compatibility(
                        {
                            "semantics": record.get("semantics", {})
                        },
                        roles[role_name],
                    )
                    semantic_evaluations[(object_id, role_name)] = semantic
                    edges.append(
                        {
                            "source": f"object:{object_id}",
                            "target": f"role:{role_name}",
                            "relation": (
                                "SEMANTICALLY_COMPATIBLE_WITH"
                            ),
                            "status": semantic["status"],
                            "evidence": semantic,
                        }
                    )

        # Unary geometry is deliberately computed for every object and role.
        # Binary geometry has two explicit strategies. The exhaustive
        # ablation evaluates every ordered pair. Production first uses RGB
        # semantics to select the role-compatible subject/object domains,
        # then evaluates only relation directions declared by the task.
        # Same-role declarations naturally support future relations such as
        # NESTABLE_IN(container, container).
        constraints_by_relation: dict[str, list[dict[str, str]]] = {}
        for constraint in self.task_requirements["constraints"]["pairwise"]:
            constraints_by_relation.setdefault(
                constraint["relation"], []
            ).append(
                {
                    "from_role": constraint["from_role"],
                    "to_role": constraint["to_role"],
                }
            )
        object_ids = sorted(self.registry["objects"])
        exhaustive = self.pairing_strategy == "exhaustive_all_pairs"
        evaluated_pair_count = 0
        skipped_pair_count = 0
        relation_evaluation_count = 0
        pairwise_started = perf_counter()
        for relation, role_pairs in sorted(
            constraints_by_relation.items()
        ):
            for source_id in object_ids:
                for target_id in object_ids:
                    if source_id == target_id:
                        continue
                    semantically_applicable_role_pairs = [
                        pair
                        for pair in role_pairs
                        if semantic_evaluations.get(
                            (source_id, pair["from_role"]), {}
                        ).get("status")
                        == "TRUE"
                        and semantic_evaluations.get(
                            (target_id, pair["to_role"]), {}
                        ).get("status")
                        == "TRUE"
                    ]
                    if not exhaustive and not semantically_applicable_role_pairs:
                        skipped_pair_count += 1
                        continue
                    evaluation = pairwise_relation_evaluation(
                        relation,
                        self.registry["objects"][source_id],
                        self.registry["objects"][target_id],
                        self.config,
                    )
                    relation_evaluation_count += 1
                    evaluated_pair_count += 1
                    role_projection_eligible = any(
                        role_evaluations[(source_id, pair["from_role"])][
                            "status"
                        ]
                        == "TRUE"
                        and role_evaluations[(target_id, pair["to_role"])][
                            "status"
                        ]
                        == "TRUE"
                        for pair in role_pairs
                    )
                    edges.append(
                        {
                            "source": f"object:{source_id}",
                            "target": f"object:{target_id}",
                            "relation": relation,
                            "status": evaluation["status"],
                            "evidence": evaluation,
                            "pairing_scope": (
                                "ALL_OBSERVED_ORDERED_OBJECT_PAIRS"
                                if exhaustive
                                else "SEMANTIC_ROLE_SCOPED_OBJECT_PAIRS"
                            ),
                            "role_binding_phase": (
                                "AFTER_PAIRWISE_GEOMETRY"
                                if exhaustive
                                else "AFTER_SEMANTIC_GATING_AND_PAIRWISE_GEOMETRY"
                            ),
                            "applicable_role_pairs": deepcopy(role_pairs),
                            "semantically_applicable_role_pairs": deepcopy(
                                semantically_applicable_role_pairs
                            ),
                            "role_projection_eligible": (
                                role_projection_eligible
                            ),
                        }
                    )
        pairwise_elapsed_seconds = perf_counter() - pairwise_started
        return {
            "schema_version": SCHEMA_VERSION,
            "inference_basis": (
                self.grounding_mode.upper().replace("-", "_")
            ),
            "grounding_mode": self.grounding_mode,
            "pairing": {
                "strategy": self.pairing_strategy,
                "scope": (
                    "ALL_OBSERVED_ORDERED_OBJECT_PAIRS"
                    if exhaustive
                    else "SEMANTIC_ROLE_SCOPED_OBJECT_PAIRS"
                ),
                "unary_geometry_scope": "ALL_OBSERVED_OBJECTS",
                "semantic_unknown_policy": (
                    "DEFER_RELATION_EVALUATION"
                ),
                "observed_object_count": len(object_ids),
                "possible_ordered_pair_count": (
                    len(object_ids) * max(0, len(object_ids) - 1)
                ),
                "evaluated_relation_pair_count": evaluated_pair_count,
                "skipped_relation_pair_count": skipped_pair_count,
                "relation_evaluation_count": relation_evaluation_count,
                "elapsed_seconds": pairwise_elapsed_seconds,
            },
            "run_id": self.run_id,
            "stage": self.next_stage,
            "nodes": nodes,
            "edges": edges,
        }

    def _render_stage(
        self,
        stage_dir: Path,
        graph: dict[str, Any],
        witness: dict[str, Any],
        stage_changes: dict[str, str],
        *,
        inspection,
        stage_evidence: dict[str, MeasurementEvidence],
    ) -> None:
        pointcloud_image = self._render_pointcloud(
            stage_changes,
            inspection=inspection,
            stage_evidence=stage_evidence,
        )
        graph_image = self._render_graph(graph, witness)
        pointcloud_image.save(stage_dir / "pointcloud.png")
        graph_image.save(stage_dir / "graph.png")

        panel_height = max(pointcloud_image.height, graph_image.height)
        summary_height = 205
        total_width = pointcloud_image.width + graph_image.width
        overview = Image.new(
            "RGB",
            (total_width, panel_height + summary_height),
            "white",
        )
        draw = ImageDraw.Draw(overview)
        status = witness["status"]
        status_color = {
            "COMPLETE": (32, 145, 72),
            "INCOMPLETE": (195, 52, 52),
            "INDETERMINATE": (215, 126, 25),
        }[status]
        draw.text(
            (24, 18),
            f"Observed-state growth · {self.scene_name} · stage {self.next_stage:03d}",
            fill=(25, 35, 50),
            font=_font(28, bold=True),
        )
        draw.rounded_rectangle(
            (24, 62, total_width - 24, summary_height - 18),
            radius=14,
            fill=(248, 250, 252),
            outline=status_color,
            width=4,
        )
        draw.text(
            (48, 78),
            f"Task: {witness['task_id']}",
            fill=(30, 38, 48),
            font=_font(22, bold=True),
        )
        draw.text(
            (total_width - 48, 78),
            status,
            anchor="ra",
            fill=status_color,
            font=_font(25, bold=True),
        )
        function_groups = witness.get("function_group_evaluations", [])
        if function_groups:
            group_label = " · ".join(
                f"{group['function_group_id']} "
                f"{group['counts']['satisfied_target_slots']}/"
                f"{group['counts']['required_target_slots']} "
                f"({group['status']})"
                for group in function_groups
            )
            distinct_label = " · ".join(
                f"{group['function_group_id']}: "
                f"{group['counts']['distinct_assigned_physical_objects']}/"
                f"{group['counts']['required_distinct_physical_objects']} "
                "distinct tools"
                for group in function_groups
            )
            draw.text(
                (48, 116),
                f"Function target slots: {group_label}",
                fill=(55, 62, 72),
                font=_font(16),
            )
            draw.text(
                (48, 146),
                f"Usage-policy counts: {distinct_label}",
                fill=(55, 62, 72),
                font=_font(16),
            )
        else:
            requirements = witness["role_requirements"]
            satisfied = witness.get("satisfied_candidate_counts")
            if satisfied is None:
                selected_witness = witness.get("selected_witness") or {}
                satisfied = {
                    role: len(selected_witness.get(role, []))
                    for role in requirements
                }
            satisfied_label = " · ".join(
                f"{role} {satisfied.get(role, 0)}/{requirements[role]}"
                for role in sorted(requirements)
            )
            missing = witness.get("missing_counts", {})
            missing_label = (
                ", ".join(
                    f"{role} ×{count}"
                    for role, count in sorted(missing.items())
                )
                or "none"
            )
            draw.text(
                (48, 116),
                f"Satisfied role counts: {satisfied_label}",
                fill=(55, 62, 72),
                font=_font(16),
            )
            draw.text(
                (48, 146),
                f"Missing role counts: {missing_label}",
                fill=(55, 62, 72),
                font=_font(16),
            )
        draw.text(
            (total_width - 48, 146),
            "Indeterminate candidates/assignments: "
            f"{sum(candidate.get('status') == 'UNKNOWN' for candidate in witness.get('candidate_evaluations', [])) + witness.get('indeterminate_assignment_count', 0)}",
            anchor="ra",
            fill=(55, 62, 72),
            font=_font(16),
        )
        overview.paste(pointcloud_image, (0, summary_height))
        overview.paste(graph_image, (pointcloud_image.width, summary_height))
        overview.save(stage_dir / "overview.png")

    def _render_pointcloud(
        self,
        stage_changes: dict[str, str],
        *,
        inspection,
        stage_evidence: dict[str, MeasurementEvidence],
    ) -> Image.Image:
        width, height = 850, 1100
        image = Image.new("RGB", (width, height), (248, 250, 252))
        draw = ImageDraw.Draw(image)
        draw.text(
            (28, 24),
            f"Stage-local evidence · {inspection.region_id}",
            fill=(25, 35, 50),
            font=_font(25, bold=True),
        )

        def project(points: np.ndarray) -> np.ndarray:
            finite = np.asarray(points, dtype=np.float64).reshape((-1, 3))
            return np.column_stack(
                (
                    0.866 * (finite[:, 0] - finite[:, 1]),
                    0.46 * (finite[:, 0] + finite[:, 1])
                    - 1.15 * finite[:, 2],
                )
            )

        volume = inspection.metadata["inspection_volume"]
        minimum = np.asarray(volume["minimum_world_m"], dtype=float)
        maximum = np.asarray(volume["maximum_world_m"], dtype=float)
        volume_corners = np.asarray(
            [
                [x, y, z]
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
                for z in (minimum[2], maximum[2])
            ]
        )
        camera_positions = np.asarray(
            [
                capture.position_world_m
                for capture in inspection.cameras.values()
            ]
        )
        target_positions = np.asarray(
            [
                capture.target_world_m
                for capture in inspection.cameras.values()
            ]
        )
        projected_sets = [
            (
                object_id,
                project(evidence.measurement_points),
                _evidence_color(object_id),
            )
            for object_id, evidence in stage_evidence.items()
            if len(evidence.measurement_points)
        ]
        rejected_sets = [
            project(cloud.points)
            for cloud in inspection.rejected_clouds.values()
            if len(cloud.points)
        ]
        # Accepted measurement evidence and the configured inspection geometry
        # define the plot.  Rejected scene points can greatly outnumber a thin
        # accepted utensil; including them in percentile bounds can clip the
        # only valid evidence completely out of the rendered frame.
        bounds_sets = [
            project(volume_corners),
            project(camera_positions),
            project(target_positions),
            *[projected for _object_id, projected, _color in projected_sets],
        ]
        combined = np.concatenate(bounds_sets)
        lower = np.min(combined, axis=0)
        upper = np.max(combined, axis=0)
        span = np.maximum(upper - lower, 1e-6)
        margin = 65
        scale = min(
            (width - 2 * margin) / span[0],
            (height - 190) / span[1],
        )

        def pixels(projected: np.ndarray) -> np.ndarray:
            result = (projected - lower) * scale
            result[:, 0] += margin
            result[:, 1] += 105
            result[:, 1] = height - 85 - result[:, 1]
            return result

        corner_pixels = pixels(project(volume_corners))
        corner_by_bits = {
            (ix, iy, iz): corner_pixels[index]
            for index, (ix, iy, iz) in enumerate(
                [
                    (ix, iy, iz)
                    for ix in (0, 1)
                    for iy in (0, 1)
                    for iz in (0, 1)
                ]
            )
        }
        for bits, start in corner_by_bits.items():
            for axis in range(3):
                if bits[axis] != 0:
                    continue
                other = list(bits)
                other[axis] = 1
                end = corner_by_bits[tuple(other)]
                self._dashed_line(
                    draw,
                    tuple(start),
                    tuple(end),
                    fill=(58, 122, 84),
                    width=2,
                    dash=7,
                )

        camera_pixels = pixels(project(camera_positions))
        target_pixels = pixels(project(target_positions))
        for index, (camera_id, capture) in enumerate(
            inspection.cameras.items()
        ):
            camera_pixel = camera_pixels[index]
            target_pixel = target_pixels[index]
            usable = bool(capture.validation.get("usable", False))
            color = (40, 90, 160) if usable else (190, 65, 65)
            draw.line(
                (*camera_pixel, *target_pixel),
                fill=color,
                width=2,
            )
            x, y = camera_pixel
            draw.polygon(
                ((x, y - 7), (x - 7, y + 6), (x + 7, y + 6)),
                fill=color,
            )
            draw.text(
                (x + 9, y - 7),
                camera_id.removeprefix("inspection_"),
                fill=color,
                font=_font(11),
            )

        for projected in rejected_sets:
            for x, y in pixels(projected):
                if 4 <= x < width - 4 and 80 <= y < height - 80:
                    draw.ellipse(
                        (x - 1, y - 1, x + 1, y + 1),
                        fill=(175, 175, 175),
                    )

        for object_id, projected, color in projected_sets:
            if len(projected) > 5000:
                projected = projected[:: math.ceil(len(projected) / 5000)]
            for x, y in pixels(projected):
                if 4 <= x < width - 4 and 80 <= y < height - 80:
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

        valid_cameras = inspection.quality["valid_camera_count"]
        draw.text(
            (28, 62),
            f"{len(stage_evidence)} accepted objects · "
            f"{len(inspection.rejected_clouds)} rejected · "
            f"{valid_cameras}/5 valid cameras",
            fill=(65, 70, 80),
            font=_font(15),
        )
        legend = [
            (object_id, color)
            for object_id, _projected, color in projected_sets
        ]
        legend.extend(
            (
                ("rejected/outside", (175, 175, 175)),
                ("inspection volume", (58, 122, 84)),
            )
        )
        for index, (label, color) in enumerate(legend):
            column = index % 3
            row = index // 3
            x = 22 + column * 275
            y = height - 66 + row * 25
            draw.rectangle(
                (x, y, x + 17, y + 17),
                fill=color,
            )
            draw.text(
                (x + 25, y - 2),
                label,
                fill=(45, 50, 58),
                font=_font(13),
            )
        return image

    @staticmethod
    def _dashed_line(
        draw: ImageDraw.ImageDraw,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        fill: tuple[int, int, int],
        width: int = 2,
        dash: int = 8,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance == 0:
            return
        ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
        position = 0.0
        while position < distance:
            segment_end = min(position + dash, distance)
            draw.line(
                (
                    x1 + ux * position,
                    y1 + uy * position,
                    x1 + ux * segment_end,
                    y1 + uy * segment_end,
                ),
                fill=fill,
                width=width,
            )
            position += 2 * dash

    def _render_graph(
        self,
        graph: dict[str, Any],
        witness: dict[str, Any],
    ) -> Image.Image:
        object_nodes = [node for node in graph["nodes"] if node["type"] == "object"]
        region_nodes = [node for node in graph["nodes"] if node["type"] == "region"]
        role_nodes = [
            node
            for node in graph["nodes"]
            if node["type"] in {"geometric_role", "functional_role"}
        ]
        rows = max(6, math.ceil(len(object_nodes) / 2))
        width, height = 1750, max(1100, 180 + rows * 160)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text(
            (28, 24),
            "Growing observed graph",
            fill=(25, 35, 50),
            font=_font(25, bold=True),
        )
        selected_candidate_edges = {
            (
                record["object_id"],
                record["role"],
            )
            for record in witness.get("selected_candidate_edges", [])
        }
        selected_pairwise_edges = {
            (
                record["relation"],
                record["from_object"],
                record["to_object"],
            )
            for record in witness.get("selected_pairwise_relations", [])
        }
        selected_objects = {
            object_id
            for selected in (witness.get("selected_witness") or {}).values()
            for object_id in selected
        }
        selected_roles = {
            role for _object_id, role in selected_candidate_edges
        }
        assignment_role_outcomes: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        for assignment in witness.get("assignment_evaluations", []):
            decision = assignment.get("decision")
            selected = assignment.get("selected_objects", {})
            if decision == "VALID":
                for role, object_ids in selected.items():
                    for object_id in object_ids:
                        assignment_role_outcomes[(object_id, role)] = {
                            "decision": "VALID",
                            "failed_relations": [],
                        }
                continue
            if decision not in {
                "REJECTED_GEOMETRY",
                "INDETERMINATE",
            }:
                continue
            relation_status = (
                "FALSE"
                if decision == "REJECTED_GEOMETRY"
                else "UNKNOWN"
            )
            for relation in assignment.get("relation_checks", []):
                if relation.get("status") != relation_status:
                    continue
                key = (
                    relation["from_object"],
                    relation["from_role"],
                )
                if (
                    assignment_role_outcomes.get(key, {}).get(
                        "decision"
                    )
                    == "VALID"
                ):
                    continue
                outcome = assignment_role_outcomes.setdefault(
                    key,
                    {
                        "decision": decision,
                        "failed_relations": [],
                    },
                )
                if relation["relation"] not in outcome[
                    "failed_relations"
                ]:
                    outcome["failed_relations"].append(
                        relation["relation"]
                    )
        object_attributes = {
            node["id"]: node["attributes"] for node in object_nodes
        }

        positions: dict[str, tuple[float, float]] = {}
        for index, node in enumerate(region_nodes):
            y = 115 + index * (height - 250) / max(1, len(region_nodes) - 1)
            positions[node["id"]] = (155, y)
        for index, node in enumerate(object_nodes):
            column = index % 2
            row = index // 2
            positions[node["id"]] = (560 + column * 470, 125 + row * 160)
        for index, node in enumerate(role_nodes):
            y = 125 + index * (height - 270) / max(1, len(role_nodes) - 1)
            positions[node["id"]] = (1540, y)

        # Draw containment and relevant pairwise geometry first. Candidate
        # role links are rendered once below from the combined semantic and
        # unary-geometry decision. Drawing the three underlying graph edges
        # separately made even a four-object graph unreadable; the complete
        # edge records remain available in graph.json.
        for edge in graph["edges"]:
            if edge["source"] not in positions or edge["target"] not in positions:
                continue
            relation = edge["relation"]
            if relation in {
                "SATISFIES_GEOMETRY",
                "SEMANTICALLY_COMPATIBLE_WITH",
                "ASSIGNED_TO_ROLE",
                "CANDIDATE_FOR",
            }:
                continue
            if relation in {"INSERTABLE_IN", "REACHES_BOTTOM"}:
                source_evaluations = object_attributes.get(
                    edge["source"], {}
                ).get("functional_role_evaluations", {})
                target_evaluations = object_attributes.get(
                    edge["target"], {}
                ).get("functional_role_evaluations", {})
                source_tool = source_evaluations.get("mixing_tool", {})
                target_container = target_evaluations.get(
                    "mixing_container", {}
                )
                if (
                    source_tool.get("geometry_gate_status") != "TRUE"
                    or target_container.get("geometry_gate_status") != "TRUE"
                ):
                    continue
            start, end = positions[edge["source"]], positions[edge["target"]]
            color = RELATION_COLORS.get(edge["status"], (120, 120, 120))
            source_object = edge["source"].removeprefix("object:")
            target_object = edge["target"].removeprefix("object:")
            selected_edge = (
                (
                    relation,
                    source_object,
                    target_object,
                )
                in selected_pairwise_edges
            )
            # Offset the two relations for one object pair so both remain
            # visible and can carry a short label.
            if relation in {"INSERTABLE_IN", "REACHES_BOTTOM"}:
                vector = np.asarray(end) - np.asarray(start)
                length = max(float(np.linalg.norm(vector)), 1.0)
                normal = np.asarray((-vector[1], vector[0])) / length
                offset = normal * (
                    -7.0 if relation == "INSERTABLE_IN" else 7.0
                )
                start = tuple(np.asarray(start) + offset)
                end = tuple(np.asarray(end) + offset)
            if selected_edge:
                draw.line(
                    (start, end),
                    fill=RELATION_COLORS["TRUE"],
                    width=6,
                )
            elif edge["status"] == "UNKNOWN":
                self._dashed_line(draw, start, end, fill=color, width=1, dash=6)
            else:
                draw.line((start, end), fill=color, width=2)
            if relation in {"INSERTABLE_IN", "REACHES_BOTTOM"}:
                midpoint = (
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                )
                relation_label = (
                    "fits" if relation == "INSERTABLE_IN" else "reaches"
                )
                label = f"{relation_label}: {edge['status'][0]}"
                font = _font(10, bold=True)
                text_box = draw.textbbox(
                    midpoint, label, anchor="mm", font=font
                )
                padded = (
                    text_box[0] - 3,
                    text_box[1] - 2,
                    text_box[2] + 3,
                    text_box[3] + 2,
                )
                draw.rounded_rectangle(
                    padded,
                    radius=3,
                    fill="white",
                    outline=color,
                    width=1,
                )
                draw.text(
                    midpoint,
                    label,
                    anchor="mm",
                    fill=color,
                    font=font,
                )

        # One composite role link per relevant object/role conveys the actual
        # gate decision without visually equating a semantic match with a
        # verified assignment.
        for node in object_nodes:
            object_id = node["attributes"]["object_id"]
            evaluations = node["attributes"].get(
                "functional_role_evaluations", {}
            )
            for role, evaluation in sorted(evaluations.items()):
                role_node_id = f"role:{role}"
                if role_node_id not in positions:
                    continue
                semantic_status = evaluation.get(
                    "semantic_gate_status", "UNKNOWN"
                )
                geometry_status = evaluation.get(
                    "geometry_gate_status", "UNKNOWN"
                )
                selected_edge = (object_id, role) in selected_candidate_edges
                relevant = (
                    selected_edge
                    or semantic_status == "TRUE"
                    or geometry_status == "TRUE"
                )
                if not relevant:
                    continue
                decision = evaluation.get("decision", "INDETERMINATE")
                assignment_outcome = assignment_role_outcomes.get(
                    (object_id, role)
                )
                if selected_edge:
                    color, width, dashed = (
                        RELATION_COLORS["TRUE"],
                        6,
                        False,
                    )
                elif (
                    assignment_outcome is not None
                    and assignment_outcome["decision"]
                    == "REJECTED_GEOMETRY"
                ) or decision in {
                    "REJECTED_SEMANTIC",
                    "REJECTED_GEOMETRY",
                }:
                    color, width, dashed = (
                        RELATION_COLORS["FALSE"],
                        2,
                        False,
                    )
                elif (
                    assignment_outcome is not None
                    and assignment_outcome["decision"]
                    == "INDETERMINATE"
                ) or decision == "INDETERMINATE":
                    color, width, dashed = (
                        RELATION_COLORS["UNKNOWN"],
                        2,
                        True,
                    )
                else:
                    color, width, dashed = ((88, 92, 150), 2, False)
                start = positions[node["id"]]
                end = positions[role_node_id]
                if dashed:
                    self._dashed_line(
                        draw,
                        start,
                        end,
                        fill=color,
                        width=width,
                        dash=7,
                    )
                else:
                    draw.line((start, end), fill=color, width=width)

        for node in region_nodes:
            x, y = positions[node["id"]]
            attrs = node["attributes"]
            inspected = attrs["inspected"]
            fill = (48, 82, 130) if inspected else (225, 227, 230)
            outline = (36, 61, 98) if inspected else (130, 130, 130)
            box = (x - 120, y - 42, x + 120, y + 42)
            if inspected:
                draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=3)
            else:
                draw.rounded_rectangle(box, radius=10, fill=fill)
                self._dashed_line(
                    draw,
                    (box[0], box[1]),
                    (box[2], box[1]),
                    fill=outline,
                    width=2,
                )
                self._dashed_line(
                    draw,
                    (box[0], box[3]),
                    (box[2], box[3]),
                    fill=outline,
                    width=2,
                )
            state = "OPEN" if attrs["open"] else "CLOSED"
            contents = attrs["contents"]
            content_label = (
                f"{len(contents)} observed"
                if isinstance(contents, list)
                else "contents UNKNOWN"
            )
            text_color = "white" if inspected else (55, 55, 55)
            draw.text(
                (x, y - 12),
                f"{attrs['region_id']} · {state}",
                anchor="mm",
                fill=text_color,
                font=_font(17, bold=True),
            )
            draw.text(
                (x, y + 14),
                content_label,
                anchor="mm",
                fill=text_color,
                font=_font(13),
            )

        for node in object_nodes:
            x, y = positions[node["id"]]
            attrs = node["attributes"]
            state = attrs.get("stage_state", "previous")
            is_selected = attrs["object_id"] in selected_objects
            fill = (
                RELATION_COLORS["TRUE"]
                if is_selected
                else STATUS_COLORS[state]
            )
            box = (x - 185, y - 68, x + 185, y + 68)
            draw.rounded_rectangle(
                box,
                radius=10,
                fill=(235, 250, 239) if is_selected else (250, 250, 250),
                outline=fill,
                width=7 if is_selected else 5,
            )
            dimensions = attrs["dimensions_m"]
            dimension_label = (
                " × ".join(
                    "?" if value is None else f"{value:.3f}"
                    for value in dimensions
                )
                + " m"
            )
            predicates = attrs.get("geometric_predicates", {})
            predicate_label = " · ".join(
                f"{name.lower()}:{predicates.get(name, {}).get('status', 'UNKNOWN')[0]}"
                for name in ("OPEN_CAVITY", "ELONGATED_OBJECT", "PLANAR_SUPPORT")
            )
            semantic = attrs.get("semantics", {}).get("validated", {})
            semantic_label = semantic.get("canonical_label") or "UNKNOWN"
            semantic_status = semantic.get("status", "UNKNOWN")
            evaluations = attrs.get(
                "functional_role_evaluations", {}
            )
            relevant_evaluations = [
                (role, evaluation)
                for role, evaluation in sorted(evaluations.items())
                if (
                    (attrs["object_id"], role) in selected_candidate_edges
                    or evaluation.get("semantic_gate_status") == "TRUE"
                    or evaluation.get("geometry_gate_status") == "TRUE"
                )
            ]
            if relevant_evaluations:
                role, evaluation = relevant_evaluations[0]
                short_role = {
                    "mixing_container": "container",
                    "mixing_tool": "tool",
                }.get(role, role)
                selected_role = (
                    attrs["object_id"], role
                ) in selected_candidate_edges
                assignment_outcome = assignment_role_outcomes.get(
                    (attrs["object_id"], role)
                )
                if selected_role:
                    decision_text = "selected witness"
                elif (
                    assignment_outcome is not None
                    and assignment_outcome["decision"]
                    == "REJECTED_GEOMETRY"
                ):
                    failed = "/".join(
                        assignment_outcome["failed_relations"]
                    )
                    decision_text = f"rejected geometry ({failed})"
                elif assignment_outcome is not None:
                    decision_text = assignment_outcome[
                        "decision"
                    ].lower()
                else:
                    decision_text = evaluation.get(
                        "decision", "?"
                    ).lower()
                decision_label = f"{short_role}: {decision_text}"
            else:
                decision_label = "role: no compatible candidate"
            lines = [
                attrs["object_id"],
                dimension_label,
                predicate_label,
                f"RGB: {semantic_label} ({semantic_status})",
                decision_label,
            ]
            for offset, line in zip((-49, -25, -1, 23, 47), lines):
                draw.text(
                    (x, y + offset),
                    line,
                    anchor="mm",
                    fill=(30, 35, 42),
                    font=_font(12, bold=offset == -49),
                )

        for node in role_nodes:
            x, y = positions[node["id"]]
            role_id = node["attributes"]["role_id"]
            box = (x - 145, y - 33, x + 145, y + 33)
            is_selected = role_id in selected_roles
            draw.rounded_rectangle(
                box,
                radius=24,
                fill=(231, 248, 235) if is_selected else (239, 230, 247),
                outline=(
                    RELATION_COLORS["TRUE"]
                    if is_selected
                    else (88, 92, 150)
                ),
                width=5 if is_selected else 3,
            )
            draw.text(
                (x, y),
                role_id,
                anchor="mm",
                fill=(45, 48, 88),
                font=_font(16, bold=True),
            )

        legend_y = height - 75
        legend = [
            ("TRUE", "solid"),
            ("FALSE", "solid"),
            ("UNKNOWN", "dashed"),
            ("valid role candidate", "geometric"),
            ("selected witness", "selected"),
        ]
        for index, (label, style) in enumerate(legend):
            x = 190 + index * 300
            color = (
                (88, 92, 150)
                if style == "geometric"
                else RELATION_COLORS["TRUE"]
                if style == "selected"
                else RELATION_COLORS[label]
            )
            if style == "dashed":
                self._dashed_line(
                    draw, (x, legend_y), (x + 65, legend_y), fill=color
                )
            else:
                draw.line((x, legend_y, x + 65, legend_y), fill=color, width=3)
            draw.text(
                (x + 78, legend_y),
                label,
                anchor="lm",
                fill=(45, 50, 58),
                font=_font(14),
            )
        return image

    def _update_growth_gif(self) -> None:
        if os.environ.get("MUJOCO_SKIP_GRAPH_MEDIA") == "1":
            return
        frames = []
        for path in sorted(self.stages_dir.glob("*/overview.png")):
            with Image.open(path) as image:
                frame = image.copy()
                frame.thumbnail((1600, 900), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (1600, 900), "white")
                canvas.paste(
                    frame,
                    ((canvas.width - frame.width) // 2, 0),
                )
                frames.append(canvas)
        if not frames:
            return
        gif_path = self.run_dir / "graph_growth.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1100,
            loop=0,
        )
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return
        video_path = self.run_dir / "graph_growth.mp4"
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(gif_path),
                    "-vf",
                    "fps=10,scale=1600:900",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            # Animation is a best-effort convenience. The immutable stage
            # images, registry, graph, and GIF remain the authoritative run
            # outputs if FFmpeg is unavailable or has no suitable encoder.
            video_path.unlink(missing_ok=True)
