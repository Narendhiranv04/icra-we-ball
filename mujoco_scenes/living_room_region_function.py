"""Integrated one-shot living-room REGION-function grounding benchmark."""

from __future__ import annotations

import itertools
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mujoco_scenes.region_ablation2 import (
    RegionAblation2Run,
    _atomic_json,
    _semantic_role,
    _tri_and,
    _value,
    evaluate_control_accessibility,
    evaluate_fits_on,
    evaluate_fits_set_on,
    evaluate_near_seat,
    load_ablation2_task,
)
from mujoco_scenes.living_room_variants import (
    PREFIX as INTEGRATED_PREFIX,
    load_living_room_variants,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG = ROOT / "configs" / "l2_integrated_region_function_task.yaml"
DEFAULT_SEMANTIC_VOCABULARY = (
    ROOT / "configs" / "l2_integrated_region_function_semantic_vocabulary.yaml"
)
DEFAULT_INTEGRATED_RIG_CONFIG = (
    ROOT / "configs" / "l2_integrated_region_function_rig.yaml"
)
PRODUCTION_MODE = "joint"
REGION_PROPOSAL_PROVENANCE = {
    "region_proposal_source": "SIMULATOR_DERIVED_NEUTRAL_SPATIAL_GATE",
    "region_proposal_encodes_function": False,
    "region_proposal_encodes_semantic_class": False,
    "region_proposal_encodes_expected_validity": False,
    "region_dimensions_for_functional_reasoning": "OBSERVED_RGBD_POINT_CLOUD",
}


EXPECTED_VARIANTS = {
    variant_id: (
        "COMPLETE" if spec["intended_outcome"] == "FEASIBLE" else "INFEASIBLE"
    )
    for variant_id, spec in load_living_room_variants().items()
}


def load_integrated_task(path: str | Path = DEFAULT_TASK_CONFIG) -> dict[str, Any]:
    config = load_ablation2_task(path)
    if config.get("requirement_entity_kind") != "REGION":
        raise ValueError("Integrated living-room requirements must be REGION-only")
    for group_id, group in config["function_groups"].items():
        if group.get("candidate_entity_kind") != "REGION":
            raise ValueError(f"Function group {group_id} is not REGION-only")
    if "object_functions" in config:
        raise ValueError("Object-function grounding is outside living-room Phase 1")
    return config


def variant_code(scene_name: str) -> str:
    if not scene_name.startswith(INTEGRATED_PREFIX):
        raise ValueError(f"Not an integrated living-room scene: {scene_name}")
    code = scene_name.removeprefix(INTEGRATED_PREFIX)
    if code not in EXPECTED_VARIANTS:
        raise ValueError(f"Unknown integrated variant: {code}")
    return code


def write_resolved_integrated_rig(scene_name: str, destination: Path) -> Path:
    """Resolve tight opaque evidence gates around the constructed supports."""
    with DEFAULT_INTEGRATED_RIG_CONFIG.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["entity_id_prefixes"] = {"seating_target": "seat"}
    config["region_proposal_provenance"] = deepcopy(
        REGION_PROPOSAL_PROVENANCE
    )
    # Import locally to retain the production/oracle dependency boundary.
    from mujoco_scenes.living_room_region_scene import build_l2_region_xml

    root = ET.fromstring(build_l2_region_xml(scene_name, robot="none"))
    support_names = (
        "a2_personal_left_top",
        "a2_personal_right_top",
        "a2_control_table_top",
    )
    selectors = list(config["region_selectors"])
    config["region_selectors"] = {}
    for selector_id, geom_name in zip(selectors, support_names):
        geom = root.find(f".//geom[@name='{geom_name}']")
        if geom is None:
            continue
        config["region_selectors"][selector_id] = {
            "candidate_rank": len(config["region_selectors"]) + 1,
            "volume": {},
        }
        center = [float(value) for value in geom.get("pos").split()]
        half = [float(value) for value in geom.get("size").split()]
        margin_xy = 0.045
        config["region_selectors"][selector_id]["volume"] = {
            "minimum_world_m": [
                center[0] - half[0] - margin_xy,
                center[1] - half[1] - margin_xy,
                center[2] - half[2] - 0.025,
            ],
            "maximum_world_m": [
                center[0] + half[0] + margin_xy,
                center[1] + half[1] + margin_xy,
                center[2] + half[2] + 0.040,
            ],
        }
    for selector_id, body_name in zip(
        config["seating_selectors"], ("a2_seat_left", "a2_seat_right")
    ):
        body = root.find(f".//body[@name='{body_name}']")
        minimum = [float("inf")] * 3
        maximum = [-float("inf")] * 3
        for geom in body.findall("geom"):
            if geom.get("type", "sphere") != "box":
                continue
            center = [float(value) for value in geom.get("pos", "0 0 0").split()]
            half = [float(value) for value in geom.get("size").split()]
            for axis in range(3):
                minimum[axis] = min(minimum[axis], center[axis] - half[axis])
                maximum[axis] = max(maximum[axis], center[axis] + half[axis])
        config["seating_selectors"][selector_id]["volume"] = {
            "minimum_world_m": [
                minimum[0] - 0.10, minimum[1] - 0.10, max(0.0, minimum[2] - 0.04)
            ],
            "maximum_world_m": [
                maximum[0] + 0.10, maximum[1] + 0.10, maximum[2] + 0.08
            ],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def _status_for_mode(row: dict[str, Any], mode: str, *, personal: bool) -> str:
    if mode == "semantic_only":
        keys = ["semantic_role_status"]
    elif mode == "geometry_only":
        keys = ["PLANAR_SUPPORT", "FITS_SET_ON" if personal else "FITS_ON"]
        keys.append("NEAR_SEAT" if personal else "ACCESSIBLE_FROM_BOTH_SEATS")
    elif mode == "joint":
        return row["compatibility_status"]
    else:
        raise ValueError(f"Unknown grounding mode: {mode}")
    return _tri_and(*(row[key] for key in keys))


class GlobalRegionAllocationSolver:
    """Exhaustive deterministic allocation over every required region slot."""

    def __init__(
        self,
        personal_rows: list[dict[str, Any]],
        shared_rows: list[dict[str, Any]],
        *,
        allow_cross_function_region_sharing: bool,
        required_personal_slot_ids: list[str] | None = None,
    ):
        self.personal_rows = personal_rows
        self.shared_rows = shared_rows
        self.allow_cross_function_region_sharing = (
            allow_cross_function_region_sharing
        )
        self.required_personal_slot_ids = required_personal_slot_ids

    @staticmethod
    def _edge_score(row: dict[str, Any]) -> float:
        margins = [
            row.get("fit_margin_m"),
            row.get("context_margin_m"),
        ]
        return sum(float(value) for value in margins if value is not None)

    def solve(self, mode: str = PRODUCTION_MODE) -> dict[str, Any]:
        # Required slots come from the fixed task contract, not from whichever
        # semantic rows happened to be observed.  Otherwise a missed payload
        # could erase a requirement and produce a vacuous COMPLETE result.
        slot_ids = (
            list(self.required_personal_slot_ids)
            if self.required_personal_slot_ids is not None
            else sorted({row["slot_id"] for row in self.personal_rows})
        )
        choices = {
            slot_id: [
                row
                for row in self.personal_rows
                if row["slot_id"] == slot_id
                and _status_for_mode(row, mode, personal=True) == "TRUE"
            ]
            for slot_id in slot_ids
        }
        shared = [
            row
            for row in self.shared_rows
            if _status_for_mode(row, mode, personal=False) == "TRUE"
        ]
        solutions = []
        products = itertools.product(*(choices[slot] for slot in slot_ids), shared)
        for selected in products:
            personal_selected = selected[:-1]
            shared_selected = selected[-1]
            personal_regions = [row["region_id"] for row in personal_selected]
            if len(set(personal_regions)) != len(personal_regions):
                continue
            if (
                not self.allow_cross_function_region_sharing
                and shared_selected["region_id"] in personal_regions
            ):
                continue
            rows = list(selected)
            solutions.append(
                {
                    "personal": list(personal_selected),
                    "shared": shared_selected,
                    "total_signed_margin": sum(
                        self._edge_score(row) for row in rows
                    ),
                    "total_candidate_rank": sum(
                        int(row["candidate_rank"]) for row in rows
                    ),
                    "region_ids": tuple(row["region_id"] for row in rows),
                }
            )
        solutions.sort(
            key=lambda item: (
                -item["total_signed_margin"],
                item["total_candidate_rank"],
                item["region_ids"],
            )
        )
        if solutions:
            selected = solutions[0]
            assignments = [
                {
                    "function_id": "PERSONAL_CUP_SAUCER_REGION",
                    "slot_id": row["slot_id"],
                    "seating_target_id": row["seating_target_id"],
                    "payload_ids": row["payload_ids"],
                    "region_id": row["region_id"],
                }
                for row in selected["personal"]
            ]
            assignments.append(
                {
                    "function_id": "SHARED_REMOTE_REGION",
                    "slot_id": "shared_remote_slot",
                    "seating_target_ids": selected["shared"][
                        "seating_target_ids"
                    ],
                    "payload_ids": selected["shared"]["payload_ids"],
                    "region_id": selected["shared"]["region_id"],
                }
            )
            return {
                "status": "COMPLETE",
                "mode": mode,
                "assignments": assignments,
                "distinct_selected_region_count": len(
                    {item["region_id"] for item in assignments}
                ),
                "complete_solution_count": len(solutions),
                "total_signed_margin": selected["total_signed_margin"],
                "cross_function_region_sharing": {
                    "allowed": self.allow_cross_function_region_sharing,
                    "satisfied": True,
                },
            }
        adjacency = {
            slot: sorted({row["region_id"] for row in rows})
            for slot, rows in choices.items()
        }
        adjacency["shared_remote_slot"] = sorted(
            {row["region_id"] for row in shared}
        )
        return {
            "status": "INFEASIBLE",
            "mode": mode,
            "assignments": [],
            "complete_solution_count": 0,
            "adjacency": adjacency,
            "uncovered_slots": [
                slot for slot, regions in adjacency.items() if not regions
            ],
            "reason": "NO_COMPLETE_GLOBAL_REGION_ALLOCATION",
            "controlled_candidate_set_fully_observed": True,
            "claim_scope": (
                "No complete verified region-functional allocation exists "
                "within the controlled fully observed candidate set."
            ),
        }

    def target_agnostic_count(self) -> dict[str, Any]:
        personal_regions = sorted(
            {
                row["region_id"]
                for row in self.personal_rows
                if row["compatibility_status"] == "TRUE"
            }
        )
        shared_regions = sorted(
            {
                row["region_id"]
                for row in self.shared_rows
                if row["compatibility_status"] == "TRUE"
            }
        )
        complete = len(personal_regions) >= 2 and bool(shared_regions)
        return {
            "status": "COMPLETE" if complete else "INFEASIBLE",
            "policy": "target_agnostic_count",
            "counted_personal_region_count": len(personal_regions),
            "shared_candidate_count": len(shared_regions),
            "decision_used_target_specific_relations": False,
            "diagnostic_only": True,
        }

    def greedy(self) -> dict[str, Any]:
        used: set[str] = set()
        assignments = []
        for slot in sorted({row["slot_id"] for row in self.personal_rows}):
            candidates = [
                row
                for row in self.personal_rows
                if row["slot_id"] == slot
                and row["compatibility_status"] == "TRUE"
                and row["region_id"] not in used
            ]
            candidates.sort(
                key=lambda row: (
                    int(row["candidate_rank"]),
                    -self._edge_score(row),
                    row["region_id"],
                )
            )
            if not candidates:
                return {
                    "status": "INFEASIBLE",
                    "policy": "greedy_target_specific",
                    "assignments": assignments,
                    "blocked_slot": slot,
                    "diagnostic_only": True,
                }
            selected = candidates[0]
            used.add(selected["region_id"])
            assignments.append(
                {"slot_id": slot, "region_id": selected["region_id"]}
            )
        shared = [
            row
            for row in self.shared_rows
            if row["compatibility_status"] == "TRUE"
            and (
                self.allow_cross_function_region_sharing
                or row["region_id"] not in used
            )
        ]
        shared.sort(
            key=lambda row: (
                int(row["candidate_rank"]),
                -self._edge_score(row),
                row["region_id"],
            )
        )
        if not shared:
            return {
                "status": "INFEASIBLE",
                "policy": "greedy_target_specific",
                "assignments": assignments,
                "blocked_slot": "shared_remote_slot",
                "diagnostic_only": True,
            }
        assignments.append(
            {
                "slot_id": "shared_remote_slot",
                "region_id": shared[0]["region_id"],
            }
        )
        return {
            "status": "COMPLETE",
            "policy": "greedy_target_specific",
            "assignments": assignments,
            "diagnostic_only": True,
        }


class IntegratedLivingRoomRegionRun(RegionAblation2Run):
    """Production region-only Phase-1 run over one INITIAL observation."""

    def __init__(self, *args, **kwargs):
        self.personal_rows: list[dict[str, Any]] = []
        self.shared_rows: list[dict[str, Any]] = []
        self.production_result: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}
        super().__init__(*args, **kwargs)
        task_path = kwargs.get("task_config", DEFAULT_TASK_CONFIG)
        self.task = load_integrated_task(task_path)

    def _write_configs(self) -> None:
        detector = {
            key: getattr(self.detector, key, None)
            for key in (
                "name", "checkpoint", "version", "device",
                "inference_size", "process_isolation",
            )
        }
        _atomic_json(
            self.run_dir / "run_config.json",
            {
                "schema_version": 1,
                "scene_name": self.scene_name,
                "variant": variant_code(self.scene_name),
                "task_id": self.task["task_id"],
                "natural_language_goal": self.task["natural_language_goal"],
                "requirement_entity_kind": "REGION",
                "production_grounding_mode": PRODUCTION_MODE,
                "production_allocation": "deterministic_exhaustive_global",
                "single_initial_observation": True,
                "perception_stage_count": 1,
                "uses_object_function_grounding": False,
                "uses_robot": False,
                "uses_foundation_model": False,
                "uses_symbolic_planning": False,
                "uses_tamp": False,
                "capture_resolution": [self.width, self.height],
                "detector": detector,
                "region_proposal_provenance": deepcopy(
                    REGION_PROPOSAL_PROVENANCE
                ),
                "created_at": datetime.now().astimezone().isoformat(),
            },
        )
        _atomic_json(self.run_dir / "task_requirements.json", self.task)

    def _build_compatibility(self) -> None:
        payloads = self._required_payloads()
        personal_group = self.task["payload_groups"]["personal_cup_saucer_sets"]
        required_count = int(personal_group["count"])
        cups = payloads.get("cup", [])
        saucers = payloads.get("saucer", [])
        if len(cups) < required_count or len(saucers) < required_count:
            bundles = []
        else:
            candidates = []
            for selected_cups in itertools.permutations(cups, required_count):
                for selected_saucers in itertools.permutations(saucers, required_count):
                    pairs = list(zip(selected_cups, selected_saucers))
                    distance = sum(
                        float(
                            np.linalg.norm(
                                np.asarray(
                                    self.payload_registry[cup][
                                        "observed_centroid_world_m"
                                    ],
                                    dtype=float,
                                )[:2]
                                - np.asarray(
                                    self.payload_registry[saucer][
                                        "observed_centroid_world_m"
                                    ],
                                    dtype=float,
                                )[:2]
                            )
                        )
                        for cup, saucer in pairs
                    )
                    canonical = tuple(sorted(tuple(sorted(pair)) for pair in pairs))
                    candidates.append((distance, canonical, pairs))
            _distance, _tie_break, bundles = min(candidates)
        bundles.sort(
            key=lambda bundle: float(
                np.mean(
                    [
                        self.payload_registry[object_id][
                            "observed_centroid_world_m"
                        ][0]
                        for object_id in bundle
                    ]
                )
            )
        )
        seats = sorted(
            self.seating_registry,
            key=lambda seat_id: float(
                self.seating_registry[seat_id]["centroid_world_m"][0]
            ),
        )
        personal_role = self.task["semantic_requirements"]["region_roles"][
            "personal_cup_saucer_region"
        ]
        shared_role = self.task["semantic_requirements"]["region_roles"][
            "shared_remote_region"
        ]
        maximum_near = float(
            self.task["geometric_requirements"]["personal_context"]
            ["maximum_centroid_distance_m"]
        )
        for slot_index, (bundle, seat_id) in enumerate(
            zip(bundles, seats), 1
        ):
            for region_id, region in self.region_registry.items():
                semantic = _semantic_role(
                    region["semantics"],
                    accepted=personal_role["accepted_categories"],
                    rejected=personal_role["rejected_categories"],
                )
                planar_value = _value(region["geometry"], "PLANAR_SUPPORT")
                planar = (
                    "UNKNOWN" if planar_value is None
                    else "TRUE" if planar_value else "FALSE"
                )
                fit = evaluate_fits_set_on(
                    [self.payload_registry[item]["geometry"] for item in bundle],
                    region["geometry"],
                    task_config=self.task,
                )
                near = evaluate_near_seat(
                    region["geometry"], self.seating_registry[seat_id],
                    maximum_distance_m=maximum_near,
                )
                target_semantic = self.seating_registry[seat_id][
                    "semantic_role"
                ]["status"]
                status = _tri_and(
                    semantic["status"], planar, fit["status"], near["status"],
                    "TRUE" if target_semantic == "TRUE" else "UNKNOWN",
                )
                self.personal_rows.append(
                    {
                        "function_id": "PERSONAL_CUP_SAUCER_REGION",
                        "slot_id": f"personal_table_slot_{slot_index}",
                        "payload_ids": list(bundle),
                        "seating_target_id": seat_id,
                        "region_id": region_id,
                        "candidate_rank": region["candidate_rank"],
                        "region_semantic_label": region["semantics"].get(
                            "canonical_label"
                        ),
                        "semantic_role_status": semantic["status"],
                        "PLANAR_SUPPORT": planar,
                        "FITS_SET_ON": fit["status"],
                        "NEAR_SEAT": near["status"],
                        "compatibility_status": status,
                        "fit_margin_m": fit.get("signed_clearance_margin_m"),
                        "context_margin_m": near.get("signed_margin_m"),
                        "fit_evidence": fit,
                        "context_evidence": near,
                        "region_evidence_path": region["provenance"][
                            "measurement_cloud_path"
                        ],
                        "payload_evidence_paths": [
                            self.payload_registry[item]["provenance"]
                            ["measurement_cloud_path"] for item in bundle
                        ],
                        "target_evidence_path": self.seating_registry[seat_id]
                        ["provenance"]["evidence_path"],
                    }
                )
        controls = payloads.get("tv_remote", [])[:1]
        if len(controls) == 1:
            for region_id, region in self.region_registry.items():
                semantic = _semantic_role(
                    region["semantics"],
                    accepted=shared_role["accepted_categories"],
                    rejected=shared_role["rejected_categories"],
                )
                planar_value = _value(region["geometry"], "PLANAR_SUPPORT")
                planar = (
                    "UNKNOWN" if planar_value is None
                    else "TRUE" if planar_value else "FALSE"
                )
                fit = evaluate_fits_on(
                    self.payload_registry[controls[0]]["geometry"],
                    region["geometry"], task_config=self.task,
                )
                access = evaluate_control_accessibility(
                    region["geometry"], self.seating_registry.values(),
                    maximum_distance_m=float(
                        self.task["geometric_requirements"]["control_context"]
                        ["maximum_distance_to_each_seat_m"]
                    ),
                )
                status = _tri_and(
                    semantic["status"], planar, fit["status"], access["status"]
                )
                self.shared_rows.append(
                    {
                        "function_id": "SHARED_REMOTE_REGION",
                        "slot_id": "shared_remote_slot",
                        "payload_ids": controls,
                        "seating_target_ids": sorted(self.seating_registry),
                        "region_id": region_id,
                        "candidate_rank": region["candidate_rank"],
                        "region_semantic_label": region["semantics"].get(
                            "canonical_label"
                        ),
                        "semantic_role_status": semantic["status"],
                        "PLANAR_SUPPORT": planar,
                        "FITS_ON": fit["status"],
                        "ACCESSIBLE_FROM_BOTH_SEATS": access["status"],
                        "compatibility_status": status,
                        "fit_margin_m": fit.get("signed_fit_margin_m"),
                        "context_margin_m": access.get("signed_margin_m"),
                        "fit_evidence": fit,
                        "context_evidence": {
                            **access,
                            "relation": "ACCESSIBLE_FROM_BOTH_SEATS",
                        },
                        "region_evidence_path": region["provenance"]
                        ["measurement_cloud_path"],
                        "payload_evidence_paths": [
                            self.payload_registry[item]["provenance"]
                            ["measurement_cloud_path"] for item in controls
                        ],
                    }
                )

    def _evaluate_policies(self) -> dict[str, Any]:
        solver = GlobalRegionAllocationSolver(
            self.personal_rows,
            self.shared_rows,
            allow_cross_function_region_sharing=bool(
                self.task["allow_cross_function_region_sharing"]
            ),
            required_personal_slot_ids=[
                "personal_table_slot_1", "personal_table_slot_2"
            ],
        )
        self.diagnostics = {
            mode: solver.solve(mode)
            for mode in ("semantic_only", "geometry_only", "joint")
        }
        self.diagnostics["target_agnostic_count"] = (
            solver.target_agnostic_count()
        )
        self.diagnostics["greedy_target_specific"] = solver.greedy()
        self.diagnostics["global_target_specific"] = deepcopy(
            self.diagnostics["joint"]
        )
        self.diagnostics["global_target_specific"]["policy"] = (
            "global_target_specific"
        )
        self.production_result = deepcopy(self.diagnostics[PRODUCTION_MODE])
        self.event(
            "GLOBAL_REGION_ALLOCATION_COMPUTED",
            status=self.production_result["status"],
            complete_solution_count=self.production_result[
                "complete_solution_count"
            ],
        )
        return self.diagnostics

    def _persist(self, semantic_timing: dict[str, Any]) -> None:
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
            "personal_cup_saucer_rows": self.personal_rows,
            "shared_remote_rows": self.shared_rows,
            "unknown_edges_admitted": False,
        }
        _atomic_json(self.run_dir / "compatibility_matrix.json", matrix)
        _atomic_json(self.run_dir / "diagnostic_modes.json", self.diagnostics)
        _atomic_json(
            self.run_dir / "predicted_feasibility.json",
            {
                "schema_version": 1,
                "status": self.production_result["status"],
                "production_mode": PRODUCTION_MODE,
                "controlled_candidate_set_fully_observed": True,
            },
        )
        assignments = self.production_result.get("assignments", [])
        selected_evidence = []
        for assignment in assignments:
            rows = (
                self.personal_rows
                if assignment["function_id"] == "PERSONAL_CUP_SAUCER_REGION"
                else self.shared_rows
            )
            row = next(
                item
                for item in rows
                if item["region_id"] == assignment["region_id"]
                and item["slot_id"] == assignment["slot_id"]
            )
            selected_evidence.append(
                {**assignment, "selected_compatibility_evidence": row}
            )
        witness = {
            "schema_version": 1,
            "witness_kind": "REGION_FUNCTIONAL_ALLOCATION",
            "task_id": self.task["task_id"],
            "status": self.production_result["status"],
            "requirement_entity_kind": "REGION",
            "functional_requirements": selected_evidence,
            "distinct_selected_region_count": self.production_result.get(
                "distinct_selected_region_count", 0
            ),
            "cross_function_region_sharing": self.production_result.get(
                "cross_function_region_sharing",
                {"allowed": False, "satisfied": False},
            ),
            "diagnostic": (
                None if self.production_result["status"] == "COMPLETE"
                else self.production_result
            ),
            "action_sequence": None,
            "symbolic_plan": None,
        }
        _atomic_json(self.run_dir / "functional_region_witness.json", witness)
        _atomic_json(
            self.run_dir / "region_assignments.json",
            {"assignments": selected_evidence},
        )
        _atomic_json(
            self.run_dir / "timings.json",
            {
                "rgbd_capture": self.observation.timings_seconds,
                "semantic_inference": semantic_timing,
            },
        )
        _atomic_json(
            self.run_dir / "evidence_manifest.json",
            {
                "single_observation": "observation/inspection_metadata.json",
                "region_registry": "region_registry.json",
                "payload_registry": "payload_registry.json",
                "seating_registry": "seating_target_registry.json",
                "compatibility_matrix": "compatibility_matrix.json",
                "semantic_overview": "observation/semantic_overview.png",
                "scene_overview": "observation/initial_scene_overview.png",
            },
        )

    def validate_expected(self) -> dict[str, Any]:
        expected = EXPECTED_VARIANTS[variant_code(self.scene_name)]
        actual = self.production_result["status"]
        result = {"expected": expected, "actual": actual, "passed": expected == actual}
        _atomic_json(self.run_dir / "expectation_validation.json", result)
        return result
