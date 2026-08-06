"""Single-observation target-specific destination-region ablation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mujoco_scenes.region_ablation2 import (
    InitialEvidenceCapture,
    InitialObservation,
    _atomic_json,
    _semantic_role,
    _tri_and,
    _value,
    evaluate_near_seat,
    evaluate_fits_set_on,
    run_initial_semantics,
)
from mujoco_scenes.region_grounding import (
    REGION_MEASUREMENT_PURPOSE,
    evaluate_fits_on,
    extract_payload_properties,
    extract_region_properties,
)
from mujoco_scenes.semantic_grounding import (
    NullSemanticDetector,
    SemanticDetector,
    load_semantic_config,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG = ROOT / "configs" / "l2_region_ablation3_task.yaml"
DEFAULT_EVALUATION_CONFIG = (
    ROOT / "configs" / "l2_region_ablation3_evaluation.yaml"
)
DEFAULT_RIG_CONFIG = ROOT / "configs" / "l2_region_ablation3_rig.yaml"
DEFAULT_SEMANTIC_VOCABULARY = (
    ROOT / "configs" / "l2_region_ablation3_semantic_vocabulary.yaml"
)
POLICIES = (
    "target_agnostic_count",
    "greedy_target_specific",
    "global_target_specific",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ablation3_task(path: str | Path = DEFAULT_TASK_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        task = yaml.safe_load(source)
    group = task["function_groups"]["personal_drinks"]
    if group["usage_policy"] != "DEDICATED_REGION_PER_TARGET":
        raise ValueError("Ablation 3 requires dedicated target regions")
    return task


def hall_diagnostics(
    targets: list[str], compatible_by_target: dict[str, set[str]]
) -> list[dict[str, Any]]:
    diagnostics = []
    for size in range(1, len(targets) + 1):
        for subset in itertools.combinations(targets, size):
            neighbours = sorted(
                set().union(*(compatible_by_target[target] for target in subset))
            )
            diagnostics.append(
                {
                    "target_subset": list(subset),
                    "neighbouring_region_ids": neighbours,
                    "target_subset_size": len(subset),
                    "neighbourhood_size": len(neighbours),
                    "coverage_deficit": max(0, len(subset) - len(neighbours)),
                    "hall_condition_satisfied": len(neighbours) >= len(subset),
                }
            )
    return diagnostics


class TargetSpecificMatcher:
    """Deterministic N-region/M-target matching over TRUE evidence edges."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        target_ids: list[str],
        required_target_count: int,
    ):
        self.rows = rows
        self.targets = sorted(target_ids)
        self.required_target_count = required_target_count
        self.true_rows = [
            row for row in rows if row["compatibility_status"] == "TRUE"
        ]
        self.edge = {
            (row["region_id"], row["seating_target_id"]): row
            for row in self.true_rows
        }
        self.regions = sorted({row["region_id"] for row in rows})
        self.general = {
            row["region_id"]: row
            for row in rows
            if row["general_suitability_status"] == "TRUE"
        }

    def _all_matchings(self) -> list[dict[str, Any]]:
        results = []

        def visit(index: int, used: set[str], assignments: list[dict]):
            if index == len(self.targets):
                rank = sum(item["candidate_rank"] for item in assignments)
                distance = sum(
                    item["near_seat_margin_m"] for item in assignments
                )
                fit = sum(item["fit_margin_m"] for item in assignments)
                results.append(
                    {
                        "assignments": deepcopy(assignments),
                        "covered_target_ids": [
                            item["seating_target_id"] for item in assignments
                        ],
                        "selected_region_ids": [
                            item["region_id"] for item in assignments
                        ],
                        "matching_cardinality": len(assignments),
                        "tie_break": {
                            "total_candidate_rank": rank,
                            "total_distance_margin_m": distance,
                            "total_fit_margin_m": fit,
                            "persistent_region_ids": sorted(used),
                            "persistent_target_ids": self.targets,
                        },
                        "ranking_key": [
                            -len(assignments),
                            rank,
                            -distance,
                            -fit,
                            sorted(used),
                            self.targets,
                        ],
                    }
                )
                return
            target = self.targets[index]
            visit(index + 1, used, assignments)
            options = sorted(
                (
                    row
                    for row in self.true_rows
                    if row["seating_target_id"] == target
                    and row["region_id"] not in used
                ),
                key=lambda row: (row["candidate_rank"], row["region_id"]),
            )
            for row in options:
                assignment = {
                    "region_id": row["region_id"],
                    "seating_target_id": target,
                    "candidate_rank": row["candidate_rank"],
                    "near_seat_margin_m": row["near_seat_margin_m"],
                    "fit_margin_m": row["fit_margin_m"],
                }
                visit(
                    index + 1,
                    used | {row["region_id"]},
                    assignments + [assignment],
                )

        visit(0, set(), [])
        results.sort(key=lambda item: item["ranking_key"])
        return results

    def global_result(self) -> dict[str, Any]:
        matchings = self._all_matchings()
        winner = matchings[0]
        cardinality = winner["matching_cardinality"]
        complete = (
            cardinality == self.required_target_count
            and len(self.targets) == self.required_target_count
        )
        compatible = {
            target: {
                row["region_id"]
                for row in self.true_rows
                if row["seating_target_id"] == target
            }
            for target in self.targets
        }
        return self._result(
            policy="global_target_specific",
            status="COMPLETE" if complete else "EXHAUSTED",
            assignments=winner["assignments"],
            maximum_matching_cardinality=cardinality,
            alternatives=[
                item
                for item in matchings[1:6]
                if item["matching_cardinality"] == cardinality
            ],
            failure_reason=None if complete else "TARGET_COVERAGE_DEFICIT",
            hall=hall_diagnostics(self.targets, compatible),
            tie_break=winner["tie_break"],
        )

    def greedy_result(self) -> dict[str, Any]:
        used, assignments = set(), []
        blocked = []
        for target in self.targets:
            options = sorted(
                (
                    row
                    for row in self.true_rows
                    if row["seating_target_id"] == target
                    and row["region_id"] not in used
                ),
                key=lambda row: (row["candidate_rank"], row["region_id"]),
            )
            if not options:
                blocked.append(target)
                continue
            row = options[0]
            used.add(row["region_id"])
            assignments.append(
                {
                    "region_id": row["region_id"],
                    "seating_target_id": target,
                    "candidate_rank": row["candidate_rank"],
                    "near_seat_margin_m": row["near_seat_margin_m"],
                    "fit_margin_m": row["fit_margin_m"],
                }
            )
        complete = (
            len(assignments) == self.required_target_count
            and not blocked
            and len(self.targets) == self.required_target_count
        )
        maximum = self.global_result()["maximum_matching_cardinality"]
        return self._result(
            policy="greedy_target_specific",
            status="COMPLETE" if complete else "EXHAUSTED",
            assignments=assignments,
            maximum_matching_cardinality=maximum,
            alternatives=[],
            failure_reason=None if complete else "GREEDY_ASSIGNMENT_BLOCKED",
            blocked_targets=blocked,
        )

    def count_result(self) -> dict[str, Any]:
        selected = sorted(
            self.general.values(),
            key=lambda row: (row["candidate_rank"], row["region_id"]),
        )[: self.required_target_count]
        complete = len(selected) >= self.required_target_count
        restricted = [
            row
            for row in self.rows
            if row["region_id"] in {item["region_id"] for item in selected}
        ]
        maximum = TargetSpecificMatcher(
            restricted,
            target_ids=self.targets,
            required_target_count=self.required_target_count,
        ).global_result()["maximum_matching_cardinality"]
        return self._result(
            policy="target_agnostic_count",
            status="COMPLETE" if complete else "EXHAUSTED",
            assignments=[],
            maximum_matching_cardinality=maximum,
            alternatives=[],
            failure_reason=None if complete else "INSUFFICIENT_GENERAL_REGIONS",
            selected_region_ids=[item["region_id"] for item in selected],
            counted_region_count=len(self.general),
            decision_used_near_seat=False,
        )

    def _result(
        self,
        *,
        policy: str,
        status: str,
        assignments: list[dict],
        maximum_matching_cardinality: int,
        alternatives: list[dict],
        failure_reason: str | None,
        **extra,
    ) -> dict[str, Any]:
        covered = sorted(
            {item["seating_target_id"] for item in assignments}
        )
        selected = extra.pop(
            "selected_region_ids",
            sorted({item["region_id"] for item in assignments}),
        )
        return {
            "policy": policy,
            "status": status,
            "candidate_region_count": len(self.general),
            "target_count": len(self.targets),
            "compatibility_edge_count": len(self.true_rows),
            "selected_region_ids": selected,
            "selected_target_ids": covered,
            "region_target_assignments": assignments,
            "covered_target_ids": covered,
            "covered_target_count": (
                maximum_matching_cardinality
                if policy == "target_agnostic_count"
                else len(covered)
            ),
            "uncovered_target_ids": sorted(set(self.targets) - set(covered)),
            "duplicate_region_violations": (
                len(selected) != len(set(selected))
            ),
            "maximum_matching_cardinality": maximum_matching_cardinality,
            "alternative_complete_matchings": alternatives,
            "failure_reason": failure_reason,
            **extra,
        }


class RegionAblation3Run:
    """Actual observation, compatibility construction, and three policies."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        scene_name: str,
        task_config: str | Path = DEFAULT_TASK_CONFIG,
        evaluation_config: str | Path = DEFAULT_EVALUATION_CONFIG,
        rig_config: str | Path = DEFAULT_RIG_CONFIG,
        semantic_detector: SemanticDetector | None = None,
        semantic_config: dict | None = None,
        width: int = 1280,
        height: int = 960,
    ):
        self.run_dir = Path(run_dir).resolve()
        if self.run_dir.exists():
            raise RuntimeError(f"Run directory already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        self.scene_name = scene_name
        self.task = load_ablation3_task(task_config)
        with Path(evaluation_config).open(encoding="utf-8") as source:
            self.evaluation = yaml.safe_load(source)
        self.rig_config = Path(rig_config)
        self.detector = semantic_detector or NullSemanticDetector()
        self.semantic_config = semantic_config or load_semantic_config(
            vocabulary_path=DEFAULT_SEMANTIC_VOCABULARY
        )
        self.width, self.height = width, height
        self.events_path = self.run_dir / "events.jsonl"
        self.observation: InitialObservation | None = None
        self.payload_registry, self.region_registry = {}, {}
        self.seating_registry, self.compatibility_rows = {}, []
        self.policy_evaluations = {}
        self._write_configs()

    def _write_configs(self) -> None:
        detector = {
            key: getattr(self.detector, key, None)
            for key in (
                "name",
                "checkpoint",
                "version",
                "device",
                "inference_size",
                "process_isolation",
            )
        }
        _atomic_json(
            self.run_dir / "run_config.json",
            {
                "schema_version": 1,
                "scene_name": self.scene_name,
                "task_id": self.task["task_id"],
                "natural_language_goal": self.task["natural_language_goal"],
                "policy_modes": list(POLICIES),
                "production_policy": "global_target_specific",
                "single_initial_observation": True,
                "uses_robot": False,
                "uses_foundation_model": False,
                "uses_tamp": False,
                "capture_resolution": [self.width, self.height],
                "detector": detector,
                "created_at": datetime.now().astimezone().isoformat(),
            },
        )
        _atomic_json(self.run_dir / "task_requirements.json", self.task)

    def event(self, name: str, **payload) -> None:
        with self.events_path.open("a", encoding="utf-8") as target:
            target.write(
                json.dumps(
                    {
                        "stage": 0,
                        "observation": "INITIAL",
                        "event": name,
                        **payload,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def _resolved_rig(self) -> Path:
        with self.rig_config.open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        variants = config.pop("scene_variants")
        config["region_selectors"] = variants[self.scene_name][
            "region_selectors"
        ]
        path = self.run_dir / "resolved_inspection_rig.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def run(self, scene) -> "RegionAblation3Run":
        observation_dir = self.run_dir / "observation"
        self.observation = InitialEvidenceCapture(
            scene,
            rig_config=self._resolved_rig(),
            task_config=self.task,
            width=self.width,
            height=self.height,
        ).capture(observation_dir)
        semantics, timing = run_initial_semantics(
            self.observation,
            detector=self.detector,
            semantic_config=self.semantic_config,
            task_config=self.task,
            observation_dir=observation_dir,
        )
        self._build_registries(semantics)
        self._build_compatibility()
        self._evaluate()
        self._persist(timing)
        return self

    def _build_registries(self, semantics: dict[str, dict]) -> None:
        assert self.observation is not None
        for object_id, record in self.observation.payloads.items():
            geometry = extract_payload_properties(record["evidence"])
            semantic = semantics[object_id]
            role = None
            if semantic.get("status") == "SUPPORTED":
                role = next(
                    (
                        role_name
                        for role_name, labels in self.task[
                            "semantic_requirements"
                        ]["payload_roles"].items()
                        if semantic.get("canonical_label") in labels
                    ),
                    None,
                )
            self.payload_registry[object_id] = {
                "identity": {"object_id": object_id, "entity_type": "drink_payload"},
                "geometry": geometry,
                "semantics": semantic,
                "semantic_payload_role": role,
                "provenance": {
                    "measurement_cloud_path": record["evidence_path"],
                    "measurement_purpose": "PAYLOAD_MEASUREMENT_EVIDENCE",
                    "point_count": record["quality"]["point_count"],
                    "contributing_camera_ids": list(
                        record["evidence"].contributing_camera_ids
                    ),
                },
            }
            self.event("PAYLOAD_OBSERVED", payload_id=object_id)
        for region_id, record in self.observation.regions.items():
            self.region_registry[region_id] = {
                "identity": {
                    "region_id": region_id,
                    "entity_type": "destination_region",
                },
                "candidate_rank": record["candidate_rank"],
                "geometry": extract_region_properties(
                    record["evidence"], task_config=self.task
                ),
                "semantics": semantics[region_id],
                "provenance": {
                    "measurement_cloud_path": record["evidence_path"],
                    "measurement_purpose": REGION_MEASUREMENT_PURPOSE,
                    "point_count": record["quality"]["point_count"],
                    "contributing_camera_ids": list(
                        record["evidence"].contributing_camera_ids
                    ),
                },
            }
            self.event("DESTINATION_REGION_OBSERVED", region_id=region_id)
        for target_id, record in self.observation.seats.items():
            semantic = semantics[target_id]
            self.seating_registry[target_id] = {
                "identity": {
                    "seating_target_id": target_id,
                    "entity_type": "seating_target",
                },
                "centroid_world_m": record["centroid_world_m"],
                "point_count": record["point_count"],
                "quality_is_valid": record["quality_is_valid"],
                "semantics": semantic,
                "semantic_role": _semantic_role(
                    semantic,
                    accepted=self.task["semantic_requirements"][
                        "seating_categories"
                    ],
                ),
                "provenance": {
                    "evidence_path": record["evidence_path"],
                    "contributing_camera_ids": record[
                        "contributing_camera_ids"
                    ],
                },
            }
            self.event("SEATING_TARGET_OBSERVED", target_id=target_id)

    def _build_compatibility(self) -> None:
        drinks = sorted(
            object_id
            for object_id, record in self.payload_registry.items()
            if record["semantic_payload_role"] == "drink"
        )
        snacks = sorted(
            object_id
            for object_id, record in self.payload_registry.items()
            if record["semantic_payload_role"] == "snack_container"
        )
        refreshment_sets = list(zip(drinks[:2], snacks[:2]))
        role = self.task["semantic_requirements"]["region_roles"][
            "personal_drink_region"
        ]
        maximum = float(
            self.task["geometric_requirements"]["personal_context"][
                "maximum_centroid_distance_m"
            ]
        )
        for region_id, region in self.region_registry.items():
            semantic = _semantic_role(
                region["semantics"],
                accepted=role["accepted_categories"],
                rejected=role["rejected_categories"],
            )
            planar_value = _value(region["geometry"], "PLANAR_SUPPORT")
            planar = (
                "UNKNOWN"
                if planar_value is None
                else "TRUE" if planar_value else "FALSE"
            )
            fits = [
                evaluate_fits_set_on(
                    [
                        self.payload_registry[object_id]["geometry"]
                        for object_id in bundle
                    ],
                    region["geometry"],
                    task_config=self.task,
                )
                for bundle in refreshment_sets
            ]
            fit_status = _tri_and(*(item["status"] for item in fits))
            fit_margin = (
                min(item["signed_clearance_margin_m"] for item in fits)
                if fits and all(
                    "signed_clearance_margin_m" in item for item in fits
                )
                else None
            )
            general = _tri_and(semantic["status"], planar, fit_status)
            for target_id, target in self.seating_registry.items():
                near = evaluate_near_seat(
                    region["geometry"],
                    target,
                    maximum_distance_m=maximum,
                )
                target_semantic = target["semantic_role"]["status"]
                status = _tri_and(
                    general,
                    near["status"],
                    "TRUE" if target_semantic == "TRUE" else "UNKNOWN",
                )
                false_reasons = [
                    name
                    for name, value in (
                        ("REGION_SEMANTICS", semantic["status"]),
                        ("PLANAR_SUPPORT", planar),
                        ("FITS_SET_ON", fit_status),
                        ("NEAR_SEAT", near["status"]),
                        ("TARGET_SEMANTICS", target_semantic),
                    )
                    if value == "FALSE"
                ]
                row = {
                    "region_id": region_id,
                    "seating_target_id": target_id,
                    "candidate_rank": region["candidate_rank"],
                    "region_semantic_label": region["semantics"].get(
                        "canonical_label"
                    ),
                    "region_semantic_confidence": region["semantics"].get(
                        "confidence"
                    ),
                    "region_semantic_supporting_views": region[
                        "semantics"
                    ].get("supporting_view_count"),
                    "semantic_role_status": semantic["status"],
                    "PLANAR_SUPPORT": planar,
                    "FITS_SET_ON": fit_status,
                    "NEAR_SEAT": near["status"],
                    "general_suitability_status": general,
                    "compatibility_status": status,
                    "rejection_reason": (
                        ",".join(false_reasons)
                        if false_reasons
                        else "INSUFFICIENT_EVIDENCE" if status == "UNKNOWN" else None
                    ),
                    "region_centroid_world_m": _value(
                        region["geometry"], "centroid_world_m"
                    ),
                    "seating_centroid_world_m": target[
                        "centroid_world_m"
                    ],
                    "measured_distance_m": near.get("measured_distance_m"),
                    "maximum_distance_m": maximum,
                    "near_seat_margin_m": near.get("signed_margin_m"),
                    "fit_margin_m": fit_margin,
                    "per_refreshment_set_fit_evidence": fits,
                    "region_contributing_camera_ids": region["provenance"][
                        "contributing_camera_ids"
                    ],
                    "target_contributing_camera_ids": target["provenance"][
                        "contributing_camera_ids"
                    ],
                    "region_evidence_path": region["provenance"][
                        "measurement_cloud_path"
                    ],
                    "target_evidence_path": target["provenance"][
                        "evidence_path"
                    ],
                }
                self.compatibility_rows.append(row)
                self.event(
                    (
                        "TARGET_COMPATIBILITY_EDGE_CREATED"
                        if status == "TRUE"
                        else "TARGET_COMPATIBILITY_EDGE_REJECTED"
                    ),
                    region_id=region_id,
                    target_id=target_id,
                    status=status,
                )

    def _evaluate(self) -> None:
        target_ids = sorted(self.seating_registry)
        required = int(
            self.task["function_groups"]["personal_drinks"][
                "required_target_count"
            ]
        )
        matcher = TargetSpecificMatcher(
            self.compatibility_rows,
            target_ids=target_ids,
            required_target_count=required,
        )
        results = {
            "target_agnostic_count": matcher.count_result(),
            "greedy_target_specific": matcher.greedy_result(),
            "global_target_specific": matcher.global_result(),
        }
        if results["target_agnostic_count"]["status"] == "COMPLETE":
            self.event(
                "TARGET_AGNOSTIC_COUNT_COMPLETE",
                counted_region_count=results["target_agnostic_count"][
                    "counted_region_count"
                ],
            )
        for assignment in results["greedy_target_specific"][
            "region_target_assignments"
        ]:
            self.event("GREEDY_ASSIGNMENT_CREATED", **assignment)
        for target_id in results["greedy_target_specific"].get(
            "blocked_targets", []
        ):
            self.event("GREEDY_ASSIGNMENT_BLOCKED", target_id=target_id)
        self.event(
            "GLOBAL_MATCHING_COMPUTED",
            maximum_matching_cardinality=results[
                "global_target_specific"
            ]["maximum_matching_cardinality"],
        )
        for target_id in results["global_target_specific"][
            "uncovered_target_ids"
        ]:
            self.event("TARGET_LEFT_UNCOVERED", target_id=target_id)
        global_complete = (
            results["global_target_specific"]["status"] == "COMPLETE"
        )
        for policy, result in results.items():
            if policy == "target_agnostic_count":
                valid = (
                    result["status"] == "COMPLETE"
                    and result["maximum_matching_cardinality"] == required
                )
                classification = (
                    "DIAGNOSTIC_COMPLETE"
                    if valid
                    else "FALSE_POSITIVE_INVALID_COMPLETE"
                    if result["status"] == "COMPLETE"
                    else "CORRECT_REJECTION"
                )
            elif policy == "greedy_target_specific":
                valid = result["status"] == "COMPLETE" or not global_complete
                classification = (
                    "CORRECT"
                    if result["status"] == "COMPLETE"
                    else "FALSE_NEGATIVE"
                    if global_complete
                    else "CORRECT_REJECTION"
                )
            else:
                valid = True
                classification = (
                    "CORRECT"
                    if result["status"] == "COMPLETE"
                    else "CORRECT_REJECTION"
                )
            result["valid_against_true_task"] = valid
            result["classification"] = classification
        self.policy_evaluations = results
        if global_complete:
            self.event("TARGET_SPECIFIC_WITNESS_COMPLETE")
            self._write_handoff(results["global_target_specific"])
        else:
            self.event("TARGET_SPECIFIC_ALLOCATION_EXHAUSTED")

    def _write_handoff(self, result: dict) -> None:
        drinks = sorted(
            object_id
            for object_id, record in self.payload_registry.items()
            if record["semantic_payload_role"] == "drink"
        )
        snacks = sorted(
            object_id
            for object_id, record in self.payload_registry.items()
            if record["semantic_payload_role"] == "snack_container"
        )
        refreshment_sets = [
            list(bundle) for bundle in zip(drinks[:2], snacks[:2])
        ]
        assignments = sorted(
            result["region_target_assignments"],
            key=lambda item: item["seating_target_id"],
        )
        mapped = []
        for bundle_ids, assignment in zip(refreshment_sets, assignments):
            row = next(
                row
                for row in self.compatibility_rows
                if row["region_id"] == assignment["region_id"]
                and row["seating_target_id"]
                == assignment["seating_target_id"]
            )
            mapped.append(
                {
                    "refreshment_set_object_ids": bundle_ids,
                    **assignment,
                    "selected_edge_evidence": row,
                }
            )
        _atomic_json(
            self.run_dir / "verified_region_allocation_handoff.json",
            {
                "schema_version": 1,
                "task_id": self.task["task_id"],
                "natural_language_goal": self.task["natural_language_goal"],
                "production_policy": "global_target_specific",
                "drink_target_region_mapping": mapped,
                "persistent_refreshment_set_ids": refreshment_sets,
                "persistent_seating_target_ids": sorted(
                    self.seating_registry
                ),
                "persistent_destination_region_ids": sorted(
                    {item["region_id"] for item in assignments}
                ),
                "destination_region_ids_distinct": len(
                    {item["region_id"] for item in assignments}
                )
                == len(assignments),
                "maximum_matching_cardinality": result[
                    "maximum_matching_cardinality"
                ],
                "target_coverage_count": len(assignments),
                "semantic_evidence": {
                    "payloads": {
                        key: value["semantics"]
                        for key, value in self.payload_registry.items()
                    },
                    "regions": {
                        key: value["semantics"]
                        for key, value in self.region_registry.items()
                    },
                    "targets": {
                        key: value["semantics"]
                        for key, value in self.seating_registry.items()
                    },
                },
                "evidence_manifest_path": "evidence_manifest.json",
                "verified": True,
                "ready_for_tamp": True,
                "placement_executed": False,
                "tamp_executed": False,
            },
        )

    def _persist(self, timing: dict) -> None:
        _atomic_json(
            self.run_dir / "payload_registry.json",
            {"schema_version": 1, "objects": self.payload_registry},
        )
        _atomic_json(
            self.run_dir / "seating_target_registry.json",
            {"schema_version": 1, "seating_targets": self.seating_registry},
        )
        _atomic_json(
            self.run_dir / "region_registry.json",
            {"schema_version": 1, "regions": self.region_registry},
        )
        matrix = {
            "schema_version": 1,
            "rows": self.compatibility_rows,
            "adjacency_by_target": {
                target: sorted(
                    row["region_id"]
                    for row in self.compatibility_rows
                    if row["seating_target_id"] == target
                    and row["compatibility_status"] == "TRUE"
                )
                for target in self.seating_registry
            },
            "adjacency_by_region": {
                region: sorted(
                    row["seating_target_id"]
                    for row in self.compatibility_rows
                    if row["region_id"] == region
                    and row["compatibility_status"] == "TRUE"
                )
                for region in self.region_registry
            },
        }
        _atomic_json(
            self.run_dir / "region_target_compatibility.json", matrix
        )
        self._write_csv(
            self.run_dir / "region_target_compatibility.csv",
            self.compatibility_rows,
        )
        _atomic_json(
            self.run_dir / "policy_evaluations.json",
            {"schema_version": 1, "policies": self.policy_evaluations},
        )
        for policy in POLICIES:
            _atomic_json(
                self.run_dir / f"{policy}_result.json",
                self.policy_evaluations[policy],
            )
        _atomic_json(
            self.run_dir / "matching_diagnostics.json",
            {
                "schema_version": 1,
                "maximum_matching_cardinality": self.policy_evaluations[
                    "global_target_specific"
                ]["maximum_matching_cardinality"],
                "hall_diagnostics": self.policy_evaluations[
                    "global_target_specific"
                ].get("hall", []),
                "policies": self.policy_evaluations,
            },
        )
        _atomic_json(
            self.run_dir / "region_assignments.json",
            {
                policy: result["region_target_assignments"]
                for policy, result in self.policy_evaluations.items()
            },
        )
        paths = sorted(
            path
            for pattern in (
                "observation/cameras/*/*.png",
                "observation/regions/*/fused.ply",
                "observation/payloads/*/fused.ply",
                "observation/seats/*/observed_points.ply",
                "observation/semantics/*.json",
            )
            for path in self.run_dir.glob(pattern)
        )
        manifest = [
            {
                "path": path.relative_to(self.run_dir).as_posix(),
                "sha256": _hash_file(path),
            }
            for path in paths
        ]
        _atomic_json(
            self.run_dir / "evidence_manifest.json",
            {"schema_version": 1, "artifacts": manifest},
        )
        summary = {
            "schema_version": 1,
            "scene_name": self.scene_name,
            "task_id": self.task["task_id"],
            "single_initial_observation": True,
            "rerendered_for_policies": False,
            "semantic_inference_repeated_for_policies": False,
            "evidence_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
            "semantic_inference_timing": timing,
            "policies": self.policy_evaluations,
        }
        _atomic_json(
            self.run_dir / "offline_region_target_ablation_evaluation.json",
            summary,
        )
        _atomic_json(self.run_dir / "region_ablation3_summary.json", summary)
        _atomic_json(
            self.run_dir / "observed_graph.json", self._build_graph()
        )
        _atomic_json(
            self.run_dir / "region_ablation3_validation.json",
            self.validate_expected(),
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        flattened = [
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        ]
        fields = sorted({key for row in flattened for key in row})
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(flattened)

    def _build_graph(self) -> dict:
        nodes = [
            *[
                {"id": f"payload:{key}", "type": "PAYLOAD", "attributes": value}
                for key, value in self.payload_registry.items()
            ],
            *[
                {"id": f"region:{key}", "type": "REGION", "attributes": value}
                for key, value in self.region_registry.items()
            ],
            *[
                {"id": f"target:{key}", "type": "TARGET", "attributes": value}
                for key, value in self.seating_registry.items()
            ],
        ]
        edges = [
            {
                "source": f"region:{row['region_id']}",
                "target": f"target:{row['seating_target_id']}",
                "type": "COMPATIBLE_WITH_TARGET",
                "status": row["compatibility_status"],
            }
            for row in self.compatibility_rows
        ]
        for assignment in self.policy_evaluations[
            "global_target_specific"
        ]["region_target_assignments"]:
            edges.append(
                {
                    "source": f"region:{assignment['region_id']}",
                    "target": f"target:{assignment['seating_target_id']}",
                    "type": "ASSIGNED_TO_TARGET",
                    "status": "TRUE",
                }
            )
        return {"schema_version": 1, "nodes": nodes, "edges": edges}

    def validate_expected(self) -> dict:
        expected = self.evaluation["scenes"][self.scene_name]["expected"]
        checks = []
        for policy, requirement in expected.items():
            observed = self.policy_evaluations[policy]
            expected_valid = requirement.get("validity")
            observed_valid = (
                "VALID"
                if observed["valid_against_true_task"]
                else "INVALID"
            )
            checks.append(
                {
                    "policy": policy,
                    "expected": requirement,
                    "observed_status": observed["status"],
                    "observed_validity": observed_valid,
                    "observed_classification": observed["classification"],
                    "passed": (
                        observed["status"] == requirement["status"]
                        and (
                            expected_valid is None
                            or observed_valid == expected_valid
                        )
                        and (
                            "classification" not in requirement
                            or requirement["classification"]
                            in observed["classification"]
                        )
                        and (
                            "counted_region_count" not in requirement
                            or observed.get("counted_region_count")
                            == requirement["counted_region_count"]
                        )
                        and (
                            "maximum_matching_cardinality" not in requirement
                            or observed["maximum_matching_cardinality"]
                            == requirement["maximum_matching_cardinality"]
                        )
                    ),
                }
            )
        return {
            "schema_version": 1,
            "scene_name": self.scene_name,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }
