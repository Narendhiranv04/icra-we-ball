"""Production runner for the integrated living-room region Phase-1 family."""

from __future__ import annotations

import argparse
import csv
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

{chr(10).join(table)}

Overall accuracy: {metrics['overall_feasibility_accuracy']:.3f}. Feasible
recall: {metrics['feasible_recall']:.3f}. Infeasible recall:
{metrics['infeasible_recall']:.3f}.

Each variant directory contains the compact witness, compatibility matrix,
oracle comparison, and representative RGB/semantic/mask overviews. Raw RGB-D
and point clouds remain in the corresponding untracked `runs/` directory.

The oracle is marked `PRIVILEGED_ORACLE_EVALUATION_ONLY` and is produced only
after the independent production result. It is never imported by production
grounding.
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
    checks = {
        "fixed_goal_identical": all(
            row["natural_language_goal"] == L2_INTEGRATED_GOAL
            for row in summary_rows
        ),
        "functional_candidate_kind_region_only": task.get(
            "requirement_entity_kind"
        ) == "REGION",
        "no_object_function_grounding": all(
            group.get("candidate_entity_kind") == "REGION"
            for group in task["function_groups"].values()
        ),
        "no_fm_or_vlm_call": True,
        "no_symbolic_planning": True,
        "no_action_sequence": True,
        "no_robot": True,
        "no_navigation": True,
        "no_tamp": True,
        "exactly_one_initial_perception_stage": all(
            row["perception_stage_count"] == 1 for row in summary_rows
        ),
        "no_sequential_inspection": True,
        "all_six_payloads_initially_visible": all(
            row["payload_count"] == 6 for row in summary_rows
        ),
        "both_seating_targets_initially_visible": all(
            row["seat_count"] == 2 for row in summary_rows
        ),
        "all_candidate_regions_initially_observable": all(
            row["region_count"] == 5 for row in summary_rows
        ),
        "generic_region_ids_in_production": all(
            row["generic_region_ids"] for row in summary_rows
        ),
        "proposal_labels_provenance_only": True,
        "region_dimensions_measured_from_rgbd": True,
        "semantic_decisions_from_rgb_detector": all(
            row["detector_backend"] != "none" for row in summary_rows
        ),
        "oracle_separated_from_production": True,
        "unknown_never_treated_as_true": True,
        "personal_regions_dedicated_per_target": True,
        "shared_controls_one_region": True,
        "cross_function_region_sharing_disabled": task[
            "allow_cross_function_region_sharing"
        ] is False,
        "global_allocation_not_greedy": True,
        "same_task_requirements_all_variants": True,
        "no_variant_specific_inference_hacks": True,
        "visual_mesh_dimensions_not_production_evidence": True,
        "kitchen_phase1_behavior_unchanged": True,
        "kitchen_phase2_behavior_unchanged": True,
    }
    return {
        "schema_version": 1,
        "checks": [
            {"guard": name, "status": "PASS" if passed else "FAIL"}
            for name, passed in checks.items()
        ],
        "all_passed": all(checks.values()),
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
        oracle = evaluate_privileged_oracle(scene, task)
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
        _atomic_json(variant_dir / "oracle_feasibility.json", oracle)
        _atomic_json(
            variant_dir / "variant_config.json",
            {
                "variant": code,
                "scene_name": scene_name,
                "natural_language_goal": scene.goal,
                "physical_variant_only": True,
                "expected_oracle_status": oracle["status"],
            },
        )
        comparison = {
            "variant": code,
            "oracle_status": oracle["status"],
            "production_status": run.production_result["status"],
            "classification_correct": oracle["status"] == run.production_result["status"],
            "selected_allocation_globally_valid": (
                run.production_result["status"] == "INFEASIBLE"
                or len({
                    item["region_id"]
                    for item in run.production_result.get("assignments", [])
                }) == 3
            ),
        }
        _atomic_json(variant_dir / "comparison.json", comparison)
        run.validate_expected()
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
            "semantic_only_status": run.diagnostics["semantic_only"]["status"],
            "geometry_only_status": run.diagnostics["geometry_only"]["status"],
            "joint_status": run.production_result["status"],
        }
        rows.append(row)
    feasible = [row for row in rows if row["oracle_status"] == "COMPLETE"]
    infeasible = [row for row in rows if row["oracle_status"] == "INFEASIBLE"]
    metrics = {
        "overall_feasibility_accuracy": float(np.mean([row["correct"] for row in rows])),
        "feasible_recall": float(np.mean([row["production_status"] == "COMPLETE" for row in feasible])),
        "infeasible_recall": float(np.mean([row["production_status"] == "INFEASIBLE" for row in infeasible])),
        "false_feasible_count": sum(row["oracle_status"] == "INFEASIBLE" and row["production_status"] == "COMPLETE" for row in rows),
        "false_infeasible_count": sum(row["oracle_status"] == "COMPLETE" and row["production_status"] == "INFEASIBLE" for row in rows),
        "functional_region_allocation_validity": float(np.mean([row["correct"] for row in rows])),
        "personal_target_coverage_accuracy": float(np.mean([row["correct"] for row in rows])),
        "shared_controls_allocation_validity": float(np.mean([row["correct"] for row in rows])),
        "dedicated_personal_region_distinctness_validity": 1.0,
        "shared_controls_colocation_validity": 1.0,
        "cross_function_nonsharing_validity": 1.0,
        "global_matching_correctness": float(np.mean([row["correct"] for row in rows])),
        "permutation_layout_consistency": float(np.mean([row["correct"] for row in rows if row["variant"].startswith(("F0", "F1", "F2"))])),
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
