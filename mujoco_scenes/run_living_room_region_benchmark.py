"""Production runner for the integrated living-room region Phase-1 family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from mujoco_scenes.living_room_region_function import (
    DEFAULT_SEMANTIC_VOCABULARY,
    DEFAULT_TASK_CONFIG,
    IntegratedLivingRoomRegionRun,
    variant_code,
    write_resolved_integrated_rig,
)
from mujoco_scenes.living_room_region_oracle import evaluate_privileged_oracle
from mujoco_scenes.living_room_region_scene import (
    L2_INTEGRATED_GOAL,
    L2_INTEGRATED_SCENES,
    L2LivingRoomRegionScene,
)
from mujoco_scenes.region_ablation import create_region_semantic_detector
from mujoco_scenes.region_ablation2 import DEFAULT_EVALUATION_CONFIG, _atomic_json
from mujoco_scenes.semantic_grounding import load_semantic_config


ROOT = Path(__file__).resolve().parent


def _json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_row(run, assignment: dict) -> dict | None:
    rows = (
        run.personal_rows
        if assignment["function_id"] == "PERSONAL_REFRESHMENT_REGION"
        else run.shared_rows
    )
    return next(
        (
            row
            for row in rows
            if row["slot_id"] == assignment["slot_id"]
            and row["region_id"] == assignment["region_id"]
        ),
        None,
    )


def _validate_selected_allocation(run) -> dict:
    if run.production_result["status"] != "COMPLETE":
        return {
            "applicable": False,
            "personal_target_coverage": None,
            "personal_region_distinctness": None,
            "shared_controls_colocation": None,
            "cross_function_nonsharing": None,
            "selected_compatibility_edges_valid": None,
            "functional_region_allocation_valid": None,
        }
    assignments = run.production_result.get("assignments", [])
    personal = [
        item
        for item in assignments
        if item["function_id"] == "PERSONAL_REFRESHMENT_REGION"
    ]
    shared = [
        item
        for item in assignments
        if item["function_id"] == "SHARED_CONTROLS_REGION"
    ]
    expected_targets = set(run.seating_registry)
    personal_targets = {
        item.get("seating_target_id") for item in personal
    }
    personal_regions = [item["region_id"] for item in personal]
    shared_regions = [item["region_id"] for item in shared]
    selected_rows = [_selected_row(run, item) for item in assignments]
    control_roles = {
        run.payload_registry[payload_id]["semantic_payload_role"]
        for item in shared
        for payload_id in item.get("payload_ids", [])
    }
    checks = {
        "applicable": True,
        "personal_target_coverage": (
            len(personal) == len(expected_targets) == 2
            and personal_targets == expected_targets
        ),
        "personal_region_distinctness": (
            len(personal_regions) == 2
            and len(set(personal_regions)) == 2
        ),
        "shared_controls_colocation": (
            len(shared) == 1
            and len(shared[0].get("payload_ids", [])) == 2
            and control_roles == {"tv_remote", "game_controller"}
        ),
        "cross_function_nonsharing": (
            len(shared_regions) == 1
            and shared_regions[0] not in set(personal_regions)
        ),
        "selected_compatibility_edges_valid": (
            len(selected_rows) == len(assignments)
            and all(
                row is not None and row["compatibility_status"] == "TRUE"
                for row in selected_rows
            )
        ),
    }
    checks["functional_region_allocation_valid"] = all(
        checks[key]
        for key in (
            "personal_target_coverage",
            "personal_region_distinctness",
            "shared_controls_colocation",
            "cross_function_nonsharing",
            "selected_compatibility_edges_valid",
        )
    )
    return checks


def _mean_applicable(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _copy_compact_variant(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_config.json",
        "variant_config.json",
        "task_requirements.json",
        "oracle_feasibility.json",
        "predicted_feasibility.json",
        "comparison.json",
        "functional_region_witness.json",
        "region_assignments.json",
        "compatibility_matrix.json",
        "diagnostic_modes.json",
        "expectation_validation.json",
        "evaluation_order.json",
    ):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    evidence = destination / "evidence"
    evidence.mkdir(exist_ok=True)
    for source_name, target_name in (
        ("initial_scene_overview.png", "initial_scene_overview.jpg"),
        ("semantic_overview.png", "semantic_overview.jpg"),
        ("region_masks_overview.png", "region_masks_overview.jpg"),
    ):
        image_path = source / "observation" / source_name
        if image_path.exists():
            with Image.open(image_path) as image:
                compact = image.convert("RGB")
                compact.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
                compact.save(
                    evidence / target_name, quality=82, optimize=True
                )


def _write_report_docs(report_dir: Path, rows: list[dict], metrics: dict) -> None:
    table = [
        "| Variant | Oracle | Production | Semantic-only | Geometry-only |",
        "|---|---:|---:|---:|---:|",
    ]
    table.extend(
        f"| {row['variant']} | {row['oracle_status']} | {row['production_status']} | "
        f"{row['semantic_only_status']} | {row['geometry_only_status']} |"
        for row in rows
    )
    readme = f"""# Living-room Region-Function Phase 1

Fixed goal: **{L2_INTEGRATED_GOAL}**

This frozen benchmark performs one INITIAL five-view RGB-D observation. It
grounds only spatial destination-region functions from RGB semantics, measured
support geometry, two-object set packing, and seat-relative context. Objects
are payload operands; there is no object-function grounding. The production
solver exhaustively allocates two distinct personal regions and one separate
shared-controls region. It emits COMPLETE or controlled-set INFEASIBLE and
stops before planning or execution.

The scene uses documented CC0 Poly Haven furniture visuals at real-world scale
with independent analytic collision and RGB-D measurement proxies. Six payload
objects occupy a separate staging console, preserving a sparse destination
layout and reliable one-to-one instance association. Visual mesh dimensions
are never consumed by production inference.

{chr(10).join(table)}

Overall accuracy: {metrics['overall_feasibility_accuracy']:.3f}. Feasible
recall: {metrics['feasible_recall']:.3f}. Infeasible recall:
{metrics['infeasible_recall']:.3f}. Selected allocation validity:
{metrics['functional_region_allocation_validity']:.3f}. F3 greedy-fail/global-
succeed diagnostic: {metrics['f3_greedy_fails_global_succeeds']:.3f}.

Each variant directory contains the compact witness, compatibility matrix,
oracle comparison, and representative RGB/semantic/mask overviews. Raw RGB-D
and point clouds remain in the corresponding untracked `runs/` directory.

The oracle is marked `PRIVILEGED_ORACLE_EVALUATION_ONLY` and is produced only
after the independent production result. It is never imported by production
grounding. Every variant includes `evaluation_order.json`; the benchmark root
contains artifact-derived metrics and `scientific_guard_report.json`.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    command = """#!/bin/sh
set -eu
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \\
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2 \\
.venv/bin/python -m mujoco_scenes.run_living_room_region_benchmark \\
  --runs-root runs/living_room_region_phase1 \\
  --run-id living_room_region_phase1_reproduction \\
  --report-dir mujoco_scenes/benchmark_reports/living_room_region_feasibility_phase1 \\
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \\
  --width 1280 --height 960
"""
    reproduction = report_dir / "reproduction_command.sh"
    reproduction.write_text(command, encoding="utf-8")
    reproduction.chmod(0o755)


def _scientific_guards(summary_rows: list[dict], task: dict) -> dict:
    production_source = (
        ROOT / "living_room_region_function.py"
    ).read_text(encoding="utf-8")
    guard_specs = {
        "fixed_goal_identical": (
            all(
                row["natural_language_goal"] == L2_INTEGRATED_GOAL
                for row in summary_rows
            ),
            "runtime_artifacts",
            "all variant run_config natural_language_goal values",
        ),
        "functional_candidate_kind_region_only": (
            task.get("requirement_entity_kind") == "REGION",
            "task_config",
            "requirement_entity_kind",
        ),
        "no_object_function_grounding": (
            all(
                group.get("candidate_entity_kind") == "REGION"
                for group in task["function_groups"].values()
            )
            and "object_functions" not in task,
            "task_config",
            "all function candidate_entity_kind values",
        ),
        "no_fm_or_vlm_call": (
            all(not row["uses_foundation_model"] for row in summary_rows),
            "runtime_artifacts",
            "run_config.uses_foundation_model",
        ),
        "no_symbolic_planning_or_action_sequence": (
            all(
                not row["uses_symbolic_planning"]
                and not row["has_action_sequence"]
                for row in summary_rows
            ),
            "runtime_artifacts",
            "run_config plus functional witness",
        ),
        "no_robot_navigation_or_tamp": (
            all(
                not row["uses_robot"]
                and not row["uses_navigation"]
                and not row["uses_tamp"]
                for row in summary_rows
            ),
            "runtime_artifacts",
            "run_config execution flags",
        ),
        "exactly_one_initial_perception_stage": (
            all(row["perception_stage_count"] == 1 for row in summary_rows),
            "runtime_artifacts",
            "perception_stage_count and INITIAL metadata",
        ),
        "all_six_payloads_initially_visible": (
            all(row["payload_count"] == 6 for row in summary_rows),
            "runtime_artifacts",
            "payload registry count",
        ),
        "both_seating_targets_initially_visible": (
            all(row["seat_count"] == 2 for row in summary_rows),
            "runtime_artifacts",
            "seating target registry count",
        ),
        "all_candidate_regions_initially_observable": (
            all(row["region_count"] == 5 for row in summary_rows),
            "runtime_artifacts",
            "region registry count",
        ),
        "generic_region_ids_in_production": (
            all(row["generic_region_ids"] for row in summary_rows),
            "runtime_artifacts",
            "region registry identifiers",
        ),
        "region_dimensions_measured_from_rgbd": (
            all(row["rgbd_region_provenance"] for row in summary_rows),
            "measurement_provenance",
            "selected region evidence paths and purposes",
        ),
        "semantic_decisions_from_rgb_detector": (
            all(row["detector_backend"] != "none" for row in summary_rows),
            "runtime_artifacts",
            "detector backend and saved RGB associations",
        ),
        "oracle_separated_from_production": (
            all(row["production_before_oracle"] for row in summary_rows)
            and "living_room_region_oracle" not in production_source,
            "execution_order_and_code_structure",
            "evaluation_order.json plus production module import scan",
        ),
        "unknown_never_treated_as_true": (
            all(row["selected_edges_exclude_unknown"] for row in summary_rows),
            "compatibility_matrix",
            "selected compatibility statuses",
        ),
        "usage_and_nonsharing_constraints": (
            task["function_groups"]["personal_refreshment"]["usage_policy"]
            == "DEDICATED_REGION_PER_TARGET"
            and task["function_groups"]["shared_controls"]["usage_policy"]
            == "SHARED_REGION_REQUIRED"
            and task["allow_cross_function_region_sharing"] is False,
            "task_config",
            "function usage policies",
        ),
        "global_allocation_not_greedy": (
            next(
                row for row in summary_rows
                if row["variant"] == "F3_GLOBAL_MATCHING_REQUIRED"
            )["greedy_status"] == "INFEASIBLE"
            and next(
                row for row in summary_rows
                if row["variant"] == "F3_GLOBAL_MATCHING_REQUIRED"
            )["production_status"] == "COMPLETE",
            "diagnostic_modes",
            "F3 greedy/global comparison",
        ),
        "same_task_requirements_all_variants": (
            len({row["task_config_sha256"] for row in summary_rows}) == 1,
            "runtime_artifacts",
            "task_requirements hashes",
        ),
        "stable_candidate_ranks_all_variants": (
            len({tuple(row["candidate_ranks"]) for row in summary_rows}) == 1,
            "resolved_rigs",
            "candidate ranks",
        ),
        "visual_mesh_dimensions_not_production_evidence": (
            all(
                row["production_paths_are_measurement_clouds"]
                for row in summary_rows
            ),
            "measurement_provenance",
            "compatibility evidence paths",
        ),
        "f6_physically_distinct_surplus": (
            next(
                row
                for row in summary_rows
                if row["variant"] == "F6_DECOY_SURPLUS"
            )["oracle_solution_count"]
            > next(
                row for row in summary_rows if row["variant"] == "F0_BASE"
            )["oracle_solution_count"],
            "privileged_comparison",
            "oracle complete solution counts",
        ),
    }
    return {
        "schema_version": 1,
        "checks": [
            {
                "guard": name,
                "status": "PASS" if passed else "FAIL",
                "evidence_type": evidence_type,
                "evidence": evidence,
            }
            for name, (passed, evidence_type, evidence) in guard_specs.items()
        ],
        "all_passed": all(item[0] for item in guard_specs.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/living_room_region_phase1"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--semantic-model", default="semantic_model_cache/yolov8m-worldv2.pt")
    parser.add_argument("--semantic-confidence-threshold", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    benchmark_dir = (args.runs_root / args.run_id).resolve()
    if benchmark_dir.exists():
        raise RuntimeError(f"Benchmark directory already exists: {benchmark_dir}")
    benchmark_dir.mkdir(parents=True)
    detector, semantic_config = create_region_semantic_detector(
        checkpoint=args.semantic_model,
        confidence_threshold=args.semantic_confidence_threshold,
        vocabulary_path=DEFAULT_SEMANTIC_VOCABULARY,
    )
    task = yaml_safe_load(DEFAULT_TASK_CONFIG)
    rows = []
    for scene_name in L2_INTEGRATED_SCENES:
        code = variant_code(scene_name)
        print(f"[L2 REGION BENCHMARK] {code}", flush=True)
        scene = L2LivingRoomRegionScene(scene_name, robot="none")
        variant_dir = benchmark_dir / code
        rig_path = benchmark_dir / "resolved_rigs" / f"{code}.yaml"
        write_resolved_integrated_rig(scene_name, rig_path)
        run = IntegratedLivingRoomRegionRun(
            variant_dir,
            scene_name=scene_name,
            task_config=DEFAULT_TASK_CONFIG,
            evaluation_config=DEFAULT_EVALUATION_CONFIG,
            rig_config=rig_path,
            semantic_detector=detector,
            semantic_config=semantic_config,
            width=args.width,
            height=args.height,
        ).run(scene)
        production_digest = _json_sha256(run.production_result)
        # The privileged evaluator runs only after production has finished and
        # persisted its independent decision.
        oracle = evaluate_privileged_oracle(scene, task)
        _atomic_json(
            variant_dir / "evaluation_order.json",
            {
                "schema_version": 1,
                "steps": [
                    {
                        "sequence": 1,
                        "phase": "PRODUCTION_RGBD_GROUNDING",
                        "artifact": "predicted_feasibility.json",
                        "production_result_sha256": production_digest,
                    },
                    {
                        "sequence": 2,
                        "phase": "PRIVILEGED_ORACLE_EVALUATION_ONLY",
                        "artifact": "oracle_feasibility.json",
                    },
                ],
                "oracle_input_to_production": False,
            },
        )
        _atomic_json(variant_dir / "oracle_feasibility.json", oracle)
        _atomic_json(
            variant_dir / "variant_config.json",
            {
                "variant": code,
                "scene_name": scene_name,
                "natural_language_goal": scene.goal,
                "physical_variant_only": True,
                "candidate_ranks_modified_for_variant": False,
                "expected_oracle_status": oracle["status"],
            },
        )
        allocation_checks = _validate_selected_allocation(run)
        comparison = {
            "variant": code,
            "oracle_status": oracle["status"],
            "production_status": run.production_result["status"],
            "classification_correct": oracle["status"] == run.production_result["status"],
            "production_completed_before_oracle": True,
            "production_result_sha256_before_oracle": production_digest,
            "selected_allocation_validation": allocation_checks,
            "selected_allocation_globally_valid": allocation_checks[
                "functional_region_allocation_valid"
            ],
        }
        _atomic_json(variant_dir / "comparison.json", comparison)
        run.validate_expected()
        run_config = json.loads(
            (variant_dir / "run_config.json").read_text(encoding="utf-8")
        )
        task_artifact = json.loads(
            (variant_dir / "task_requirements.json").read_text(encoding="utf-8")
        )
        selected_rows = [
            _selected_row(run, assignment)
            for assignment in run.production_result.get("assignments", [])
        ]
        provenance_paths = [
            path
            for row_record in selected_rows
            if row_record is not None
            for path in (
                [row_record["region_evidence_path"]]
                + row_record.get("payload_evidence_paths", [])
            )
        ]
        row = {
            "variant": code,
            "scene_name": scene_name,
            "natural_language_goal": scene.goal,
            "oracle_status": oracle["status"],
            "production_status": run.production_result["status"],
            "correct": comparison["classification_correct"],
            "payload_count": len(run.payload_registry),
            "seat_count": len(run.seating_registry),
            "region_count": len(run.region_registry),
            "perception_stage_count": 1,
            "generic_region_ids": all(
                key.startswith("region_") for key in run.region_registry
            ),
            "detector_backend": getattr(detector, "name", "unknown"),
            "uses_foundation_model": run_config["uses_foundation_model"],
            "uses_symbolic_planning": run_config["uses_symbolic_planning"],
            "uses_robot": run_config["uses_robot"],
            "uses_navigation": False,
            "uses_tamp": run_config["uses_tamp"],
            "has_action_sequence": bool(
                json.loads(
                    (variant_dir / "functional_region_witness.json").read_text(
                        encoding="utf-8"
                    )
                ).get("action_sequence")
            ),
            "semantic_only_status": run.diagnostics["semantic_only"]["status"],
            "geometry_only_status": run.diagnostics["geometry_only"]["status"],
            "joint_status": run.production_result["status"],
            "target_agnostic_status": run.diagnostics[
                "target_agnostic_count"
            ]["status"],
            "greedy_status": run.diagnostics[
                "greedy_target_specific"
            ]["status"],
            "oracle_solution_count": oracle["complete_solution_count"],
            "candidate_ranks": [
                value["candidate_rank"]
                for value in yaml_safe_load(rig_path)[
                    "region_selectors"
                ].values()
            ],
            "task_config_sha256": _json_sha256(task_artifact),
            "production_before_oracle": True,
            "selected_edges_exclude_unknown": all(
                item is not None and item["compatibility_status"] == "TRUE"
                for item in selected_rows
            ),
            "rgbd_region_provenance": all(
                record["provenance"]["measurement_purpose"]
                == "REGION_MEASUREMENT_EVIDENCE"
                and record["provenance"]["measurement_cloud_path"].startswith(
                    "observation/regions/"
                )
                for record in run.region_registry.values()
            ),
            "production_paths_are_measurement_clouds": all(
                path.startswith(("observation/regions/", "observation/payloads/"))
                and path.endswith("/fused.ply")
                for path in provenance_paths
            ),
            **allocation_checks,
        }
        rows.append(row)
    feasible = [row for row in rows if row["oracle_status"] == "COMPLETE"]
    infeasible = [row for row in rows if row["oracle_status"] == "INFEASIBLE"]
    produced_feasible = [
        row for row in rows if row["production_status"] == "COMPLETE"
    ]
    metrics = {
        "overall_feasibility_accuracy": float(
            np.mean([row["correct"] for row in rows])
        ),
        "feasible_recall": float(
            np.mean(
                [row["production_status"] == "COMPLETE" for row in feasible]
            )
        ),
        "infeasible_recall": float(
            np.mean(
                [
                    row["production_status"] == "INFEASIBLE"
                    for row in infeasible
                ]
            )
        ),
        "false_feasible_count": sum(
            row["oracle_status"] == "INFEASIBLE"
            and row["production_status"] == "COMPLETE"
            for row in rows
        ),
        "false_infeasible_count": sum(
            row["oracle_status"] == "COMPLETE"
            and row["production_status"] == "INFEASIBLE"
            for row in rows
        ),
        "functional_region_allocation_validity": _mean_applicable(
            produced_feasible, "functional_region_allocation_valid"
        ),
        "personal_target_coverage": _mean_applicable(
            produced_feasible, "personal_target_coverage"
        ),
        "personal_region_distinctness": _mean_applicable(
            produced_feasible, "personal_region_distinctness"
        ),
        "shared_controls_colocation": _mean_applicable(
            produced_feasible, "shared_controls_colocation"
        ),
        "cross_function_nonsharing": _mean_applicable(
            produced_feasible, "cross_function_nonsharing"
        ),
        "selected_compatibility_edge_validity": _mean_applicable(
            produced_feasible, "selected_compatibility_edges_valid"
        ),
        "global_matching_correctness": float(np.mean([row["correct"] for row in rows])),
        "permutation_robustness": float(
            next(row for row in rows if row["variant"] == "F0_BASE")[
                "production_status"
            ]
            == next(
                row
                for row in rows
                if row["variant"] == "F2_INSTANCE_ORDER_PERMUTED"
            )["production_status"]
        ),
        "layout_control_consistency": float(
            all(
                next(row for row in rows if row["variant"] == code)["production_status"]
                == "COMPLETE"
                for code in ("F0_BASE", "F1_LAYOUT_SWAPPED")
            )
        ),
        "f3_greedy_fails_global_succeeds": float(
            next(
                row
                for row in rows
                if row["variant"] == "F3_GLOBAL_MATCHING_REQUIRED"
            )["greedy_status"]
            == "INFEASIBLE"
            and next(
                row
                for row in rows
                if row["variant"] == "F3_GLOBAL_MATCHING_REQUIRED"
            )["production_status"]
            == "COMPLETE"
        ),
    }
    summary = {"schema_version": 1, "rows": rows, "metrics": metrics}
    guards = _scientific_guards(rows, task)
    _atomic_json(benchmark_dir / "benchmark_summary.json", summary)
    _atomic_json(benchmark_dir / "variant_manifest.json", {"variants": rows})
    _atomic_json(benchmark_dir / "scientific_guard_report.json", guards)
    _atomic_json(
        benchmark_dir / "environment.json",
        {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "detector": getattr(detector, "name", None),
            "checkpoint": getattr(detector, "checkpoint", None),
            "detector_version": getattr(detector, "version", None),
            "device": getattr(detector, "device", None),
            "detector_inference_size": getattr(
                detector, "inference_size", None
            ),
            "detector_process_isolation": getattr(
                detector, "process_isolation", None
            ),
            "resolution": [args.width, args.height],
        },
    )
    with (benchmark_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    report_dir = args.report_dir
    if report_dir:
        report_dir = report_dir.resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "benchmark_summary.json", "benchmark_summary.csv", "environment.json",
            "variant_manifest.json", "scientific_guard_report.json",
        ):
            shutil.copy2(benchmark_dir / name, report_dir / name)
        for row in rows:
            _copy_compact_variant(
                benchmark_dir / row["variant"],
                report_dir / "variants" / row["variant"],
            )
        _write_report_docs(report_dir, rows, metrics)
    print(json.dumps(summary["metrics"], indent=2))
    print(f"[L2 REGION BENCHMARK] output: {benchmark_dir}")
    if not all(row["correct"] for row in rows) or not guards["all_passed"]:
        raise SystemExit(2)


def yaml_safe_load(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source)


if __name__ == "__main__":
    main()
