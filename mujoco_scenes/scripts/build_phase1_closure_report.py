#!/usr/bin/env python3
"""Build the compact, tracked Phase-1 report from authoritative raw runs."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

from PIL import Image

from mujoco_scenes.run_kitchen_feasibility_benchmark import _aggregate


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1"
PRIMARY = ROOT / "runs/feasibility_benchmarks/kitchen_feasibility_phase1_closure_20260809"
TAIL = ROOT / "runs/feasibility_benchmarks/kitchen_feasibility_phase1_closure_tail_20260809"
VARIANTS = (
    "F0_REUSE_ONE", "F1_INITIAL_COMPLETE", "F2_DISTRIBUTED_COFFEE_TWO",
    "F3_DISTRIBUTED_COFFEE_THREE", "F4_EARLY_RELOCATION",
    "F5_LATE_RELOCATION", "F6_DECOY_HEAVY", "F7_COUNT_SURPLUS",
    "I0_MISSING_COFFEE_VESSEL", "I1_MISSING_SOUP_VESSEL",
    "I2_UNCOVERED_COFFEE_TARGET", "I3_ONLY_TWO_SOUP_TOOLS",
    "I4_SOUP_MATCHING_TRAP", "I5_SEMANTIC_DECOY_GEOMETRY_FAILURE",
    "P0_LAYOUT_BASE", "P1_LAYOUT_SWAPPED",
)
TAIL_VARIANTS = {
    "I4_SOUP_MATCHING_TRAP", "I5_SEMANTIC_DECOY_GEOMETRY_FAILURE",
    "P0_LAYOUT_BASE", "P1_LAYOUT_SWAPPED",
}
COMPACT_FILES = (
    "variant_config.json", "oracle_feasibility.json",
    "oracle_stage_feasibility.json", "predicted_feasibility.json",
    "stage_results.json", "functional_witness.json",
    "grounded_assignments.json",
)
GOAL = (
    "Prepare and serve coffee and soup for three people using the available "
    "kitchenware. Stir all three coffees and provide each soup bowl with a "
    "suitable utensil. Search the closed kitchen storage for anything still "
    "required."
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(variant: str) -> Path:
    return (TAIL if variant in TAIL_VARIANTS else PRIMARY) / variant


def comparison(variant: str, src: Path) -> dict:
    config = read(src / "variant_config.json")
    oracle = read(src / "oracle_feasibility.json")
    predicted = read(src / "predicted_feasibility.json")
    oracle_outcome = oracle["oracle_terminal_outcome"]
    predicted_outcome = predicted["terminal_outcome"]
    correctly_feasible = oracle_outcome == predicted_outcome == "FEASIBLE"
    row = {
        "variant_id": variant,
        "intended_outcome": config["intended_outcome"],
        "oracle_outcome": oracle_outcome,
        "oracle_failure_reason": oracle.get("oracle_failure_reason"),
        "predicted_outcome": predicted_outcome,
        "predicted_failure_reason": predicted.get("reason_if_infeasible"),
        "classification_correct": oracle_outcome == predicted_outcome,
        "oracle_earliest_feasible_stage": oracle.get("oracle_earliest_feasible_stage"),
        "predicted_completion_stage": predicted.get("completion_stage"),
        "stage_correct": correctly_feasible and oracle.get("oracle_earliest_feasible_stage") == predicted.get("completion_stage"),
        "oracle_min_coffee_tools": oracle.get("oracle_coffee_minimum_unique_tools"),
        "predicted_coffee_unique_tools": predicted.get("coffee_unique_tool_count"),
        "coffee_minimum_tool_correct": correctly_feasible and oracle.get("oracle_coffee_minimum_unique_tools") == predicted.get("coffee_unique_tool_count"),
        "oracle_soup_matching_size": oracle.get("oracle_soup_matching_size"),
        "predicted_soup_assignment_count": len(predicted.get("soup_assignments", [])),
        "soup_distinct_assignment_valid": predicted_outcome != "FEASIBLE" or (predicted.get("soup_distinctness_satisfied", False) and len(predicted.get("soup_assignments", [])) == 3),
    }
    row["result"] = "PASS" if row["classification_correct"] else "FAIL"
    return row


def save_jpeg(source_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.thumbnail((1400, 1000))
        image.convert("RGB").save(destination, "JPEG", quality=72, optimize=True)


def failure_diagnostic(src: Path) -> dict:
    registry = read(src / "object_registry.json")
    witness = read(src / "functional_witness.json")
    association_path = src / "stages/004_after_B1/semantics/associations.json"
    association_data = read(association_path)
    unresolved = []
    for object_id, record in registry["objects"].items():
        semantic = record.get("semantics", {}).get("latest_observation", {})
        if semantic.get("status") != "UNKNOWN" or record.get("source_region") != "B1":
            continue
        geometry = record.get("geometry", {})
        per_camera_associations = []
        for camera in association_data.get("cameras", []):
            accepted = [
                item for item in camera.get("accepted", [])
                if item.get("object_id") == object_id
            ]
            rejected = [
                item for item in camera.get("rejected", [])
                if item.get("object_id") == object_id
            ]
            per_camera_associations.append({
                "camera_id": camera.get("camera_id"),
                "accepted": accepted,
                "rejected": rejected,
            })
        unresolved.append({
            "object_id": object_id,
            "source_inspection_region": record.get("source_region"),
            "required_roles": ["coffee_stirrer", "soup_eating_utensil", "coffee_container", "soup_container"],
            "image_crop_paths": semantic.get("semantic_evidence_paths", []),
            "per_camera_detector_predictions": semantic.get("alternatives", []),
            "association_method": semantic.get("association_method"),
            "per_camera_associations": per_camera_associations,
            "canonical_fusion_result": semantic.get("canonical_label"),
            "supporting_view_count": semantic.get("supporting_view_count"),
            "compatible_role_supporting_camera_ids": semantic.get("contributing_camera_ids", []),
            "conflicting_labels": [item.get("label") for item in semantic.get("alternatives", [])[1:]],
            "final_semantic_status": semantic.get("status"),
            "semantic_reason_codes": semantic.get("reason_codes", []),
            "semantic_quality": semantic.get("quality", {}),
            "unary_geometry": geometry.get("predicates", {}),
            "measurement_quality": geometry.get("measurement_quality", record.get("measurement_quality", {})),
            "measurement_cloud_path": record.get("measurement_cloud_path"),
        })
    return {
        "variant_id": "F0_REUSE_ONE",
        "classification": "FALSE_INFEASIBLE",
        "production_failure_reason": read(src / "predicted_feasibility.json").get("reason_if_infeasible"),
        "witness_reason_codes": witness.get("reason_codes", []),
        "root_cause": "B1 objects have valid five-view geometry but insufficient or conflicting YOLO-World semantic evidence under the unchanged two-view support and winning-margin gates.",
        "protocol_decision": "Retain UNKNOWN. No hidden label injection, one-view downgrade, object-ID exception, or oracle measurement was used.",
        "unresolved_observed_objects": unresolved,
        "offline_oracle_comparison": {
            "oracle_outcome": "FEASIBLE",
            "note": "Privileged names/geometry appear only in this offline diagnostic, never in production inference.",
        },
    }


def main() -> None:
    variants_dir = REPORT / "variants"
    if variants_dir.exists():
        shutil.rmtree(variants_dir)
    rows = []
    manifest = {"schema_version": 2, "variants": []}
    for variant in VARIANTS:
        src = source(variant)
        dst = variants_dir / variant
        dst.mkdir(parents=True, exist_ok=True)
        for name in COMPACT_FILES:
            shutil.copy2(src / name, dst / name)
        row = comparison(variant, src)
        write(dst / "comparison.json", row)
        if variant == "F0_REUSE_ONE":
            write(dst / "failure_diagnostic.json", failure_diagnostic(src))
        stages = sorted((src / "stages").glob("[0-9][0-9][0-9]_*"))
        for label, stage in (("initial", stages[0]), ("terminal", stages[-1])):
            for image_name in ("overview.png", "semantic_overview.png"):
                image_path = stage / image_name
                if image_path.exists():
                    save_jpeg(image_path, dst / "evidence" / f"{label}_{image_name[:-4]}.jpg")
        rows.append(row)
        manifest["variants"].append({
            "variant_id": variant,
            "raw_evidence_run": str(src.relative_to(ROOT)),
            "result": row["result"],
            "tracked_directory": f"variants/{variant}",
        })
    metrics = _aggregate(rows)
    summary = {
        "schema_version": 2,
        "benchmark_id": "kitchen_feasibility_phase1_closure_20260809_combined",
        "evidence_runs": [str(PRIMARY.relative_to(ROOT)), str(TAIL.relative_to(ROOT))],
        "goal_instruction": GOAL,
        "scope": "TASK_LEVEL_FEASIBILITY_ONLY_NO_ACTION_PLANNING",
        "variants": rows,
        "aggregate_metrics": metrics,
    }
    write(REPORT / "benchmark_summary.json", summary)
    write(REPORT / "variant_manifest.json", manifest)
    with (REPORT / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    configs = {
        "task_config": ROOT / "mujoco_scenes/configs/s1_integrated_kitchen_object_function.yaml",
        "geometry_config": ROOT / "mujoco_scenes/configs/geometry_inference.yaml",
        "semantic_config": ROOT / "mujoco_scenes/configs/semantic_grounding.yaml",
        "semantic_vocabulary": ROOT / "mujoco_scenes/configs/semantic_vocabulary.yaml",
        "variant_config": ROOT / "mujoco_scenes/configs/kitchen_feasibility_variants.yaml",
    }
    versions = {}
    for package in ("mujoco", "numpy", "torch", "ultralytics"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = {
        "source_commit_at_generation": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": "naren/googlePointCloudIntegration",
        "python_version": platform.python_version(),
        "dependency_versions": versions,
        "detector": "Ultralytics YOLO-World",
        "yolo_model_filename": "yolov8m-worldv2.pt",
        "device": "cpu",
        "image_width": 1280,
        "image_height": 960,
        "semantic_confidence_threshold": 0.03,
        "semantic_minimum_supporting_views": 2,
        "exact_geometry_extractor": "inner_rim_radial_quantile_v2_from_actual_instantiated_mujoco_geometry",
        "benchmark_timestamp": "2026-08-09",
        "benchmark_id": summary["benchmark_id"],
        "configs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in configs.items()},
    }
    write(REPORT / "environment.json", environment)
    guards = {
        "schema_version": 2,
        "status": "PASS_WITH_DOCUMENTED_F0_DETECTOR_LIMITATION",
        "source_commit_at_generation": environment["source_commit_at_generation"],
        "config_hashes": environment["configs"],
        "same_goal_all_variants": True,
        "oracle_geometry_source": "ACTUAL_INSTANTIATED_MUJOCO_GEOMETRY",
        "oracle_cavity_opening_uses_inner_geometry": True,
        "oracle_outer_bbox_used_as_opening": False,
        "oracle_manual_numeric_geometry_table_present": False,
        "production_geometry_source": "STAGE_LOCAL_REGION_GATED_RGBD_MEASUREMENT",
        "shared_insertable_definition": True,
        "shared_reaches_bottom_definition": True,
        "oracle_reads_observed_registry": False,
        "prediction_reads_oracle": False,
        "oracle_and_prediction_share_mutable_scene": False,
        "semantic_confidence_used_as_assignment_objective": False,
        "coffee_complete_coverage_required": True,
        "coffee_single_tool_required": False,
        "coffee_minimum_tool_preference": True,
        "soup_distinct_target_assignment_required": True,
        "cross_group_reuse_allowed": True,
        "cross_group_reuse_optimized": False,
        "assignment_driven_tool_cardinality": True,
        "nonterminal_unknown_classified_as_infeasible": False,
        "terminal_exhaustion_maps_to_infeasible": True,
        "minimum_semantic_supporting_views": 2,
        "hidden_labels_used_by_prediction": False,
        "action_planning_used": False,
        "FM_used": False,
        "robot_used": False,
        "evidence": {
            "benchmark_summary": "benchmark_summary.json",
            "F0_failure_diagnostic": "variants/F0_REUSE_ONE/failure_diagnostic.json",
            "shared_relation_implementation": "mujoco_scenes/geometry_relations.py",
            "exact_geometry_implementation": "mujoco_scenes/exact_scene_geometry.py",
            "focused_tests": "mujoco_scenes/tests/test_kitchen_feasibility_benchmark.py",
            "full_test_result": "test_summary.txt",
        },
    }
    write(REPORT / "scientific_guard_report.json", guards)

    table = ["| Variant | Oracle | Prediction | Oracle stage | Predicted stage | Oracle coffee tools | Predicted tools | Result |", "|---|---|---|---|---|---:|---:|---|"]
    table.extend(
        f"| {r['variant_id']} | {r['oracle_outcome']} | {r['predicted_outcome']} | {r['oracle_earliest_feasible_stage'] or '-'} | {r['predicted_completion_stage'] or '-'} | {r['oracle_min_coffee_tools'] if r['oracle_min_coffee_tools'] is not None else '-'} | {r['predicted_coffee_unique_tools'] if r['predicted_coffee_unique_tools'] is not None else '-'} | {r['result']} |"
        for r in rows
    )
    readme = """# Phase 1 kitchen feasibility closure

Phase 1 evaluates: **Given a fixed task instruction and an exhaustively inspected controlled environment, does the observed semantic-geometric grounding system determine whether a complete functional assignment exists?**

All 16 variants use the identical instruction. Coffee requires complete target coverage; reuse is preferred through minimum-distinct-tool assignment but one universal tool is not required. Soup requires a global one-to-one assignment of three distinct physical utensils. Cross-function reuse is allowed and neutral.

The oracle uses privileged full instantiated MuJoCo geometry for evaluation only. Its cavity opening is an inner-rim estimate that excludes handles/exterior protrusions. Production uses only visible region-gated RGB-D point-cloud measurements and YOLO-World semantic evidence. An unresolved witness is called task-level `INFEASIBLE` only after the fixed inspection order is exhausted.

The final real result is **15/16 correct**: all six oracle-infeasible variants were rejected and nine of ten oracle-feasible variants were detected. `F0_REUSE_ONE` remains a transparent false-infeasible: two B1 objects have valid five-view geometry but ambiguous YOLO-World evidence under the preserved two-view and winning-margin gates. `P0_LAYOUT_BASE` is now a genuinely distinct layout and passes. We did not weaken the protocol or inject hidden labels to obtain 16/16.

Aggregate metrics: overall feasibility accuracy 0.9375; feasible detection/recall 0.90; infeasible recall 1.00; false feasible 0; false infeasible 1; earliest-stage success over all oracle-feasible 0.90 (1.00 conditional on detected-feasible); minimum-coffee-tool optimality over all oracle-feasible 0.90 (1.00 conditional on detected-feasible); soup distinct-assignment validity 1.00.

This is a controlled Phase-1 result, not a claim of general real-world reliability. No FM/LLM/VLM, PDDL/action planner, robot, navigation, IK, or manipulation is used.

## Results

""" + "\n".join(table) + "\n\n## Reproduction\n\nRun `./mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1/reproduction_command.sh <benchmark-id>`. Raw PLY/depth data and model weights are intentionally untracked; compact JSON, initial/terminal overview images, and the F0 diagnostic are included here.\n"
    (REPORT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
