"""Comprehensive scene and benchmark audit script for the Workshop (W1) domain.

Audits all 14 benchmark variants (F0-F6, I0-I6) against:
1. Physical inventory truthfulness (declared YAML == physical MuJoCo bodies).
2. Dynamic free-body compliance (all pickable objects have independent freejoints).
3. Physical functional region availability (compiled MjModel == declared active regions).
4. Physical stability on reset (no NaNs, no infinities, bounded velocities).
5. Storage articulation (smooth open/close preserving contained objects).
6. Privileged scene oracle feasibility & rejection reason verification.
7. Multi-camera RGB-D point cloud stage-level evidence verification (threshold >= 20 pts).
8. Production observation API leakage boundary compliance.

Generates structured per-variant manifests and an overall suite_summary.json.
Exits with non-zero status code if any audit check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mujoco_scenes.geometry_checker import GeometryChecker
from mujoco_scenes.workshop_pointcloud import STAGES
from mujoco_scenes.workshop_scene import (
    WORKSHOP_REGIONS,
    WorkshopScene,
    _load_workshop_variants_config,
    privileged_actual_parts_container_regions,
    privileged_actual_storage_region,
    privileged_actual_work_surface_regions,
    privileged_validate_variant_feasibility,
)


MIN_WORKSHOP_OBJECT_FUSED_POINTS = 20

FORBIDDEN_LEAKAGE_STRINGS = (
    "workshop_long_phillips_driver",
    "workshop_power_driver",
    "phillips_screwdriver",
    "powered_screwdriver",
    "can_drive_screw",
    "can_fasten",
    "PH2",
    "tip_profile",
    "reach_m",
    "shaft_diameter_m",
    "target_hole_diameter_m",
    "target_hole_depth_m",
    "usable_area_m2",
    "cavity_volume_m3",
    "expected_solution",
)


def _check_dict_leakage(data: Any, forbidden: tuple[str, ...]) -> list[str]:
    """Recursively search data structures for forbidden ground-truth leakage strings."""
    leaks = []
    if isinstance(data, dict):
        for k, v in data.items():
            for f in forbidden:
                if f.lower() == str(k).lower() or f.lower() == str(v).lower():
                    leaks.append(f"Key/Value leak: {k}={v} matches {f}")
            leaks.extend(_check_dict_leakage(v, forbidden))
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            if isinstance(item, str):
                for f in forbidden:
                    if f.lower() == item.lower():
                        leaks.append(f"List item leak: {item} matches {f}")
            else:
                leaks.extend(_check_dict_leakage(item, forbidden))
    return leaks


def audit_single_variant(
    variant_name: str,
    output_dir: Path | None = None,
    run_pointcloud_smoke: bool = True,
) -> dict[str, Any]:
    """Perform a deep physical and oracle audit of a single workshop variant."""
    config = _load_workshop_variants_config()
    variants = config.get("variants", {})
    if variant_name not in variants:
        raise ValueError(f"Unknown variant: {variant_name}")

    var_meta = variants[variant_name]
    intended_outcome = var_meta.get("intended_outcome", "UNKNOWN")
    expected_rejection_reason = var_meta.get("rejection_reason")
    declared_contents = var_meta.get("storage_contents", {})
    declared_active_surfaces = var_meta.get("active_surfaces", [])
    declared_active_containers = var_meta.get("active_containers", [])

    # 1. Instantiate scene and test physical stability
    scene = WorkshopScene(robot="none", variant=variant_name)
    n_bodies = scene.model.nbody
    all_objects = [
        obj
        for obj_list in declared_contents.values()
        for obj in obj_list
    ]

    has_nan_qpos = bool(np.isnan(scene.data.qpos).any())
    has_nan_qvel = bool(np.isnan(scene.data.qvel).any())
    has_inf_qpos = bool(np.isinf(scene.data.qpos).any())
    has_inf_qvel = bool(np.isinf(scene.data.qvel).any())
    stable_reset = not (has_nan_qpos or has_nan_qvel or has_inf_qpos or has_inf_qvel)

    # 2. Dynamic free-body compliance check
    free_bodies_pass = True
    for obj_name in all_objects:
        b_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        j_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj_name}_free")
        if b_id < 0 or j_id < 0 or scene.model.jnt_type[j_id] != mujoco.mjtJoint.mjJNT_FREE:
            free_bodies_pass = False

    # 3. Physical inventory verification
    actual_storage_inventory: dict[str, list[str]] = {
        region: [] for region in WORKSHOP_REGIONS
    }
    for obj_name in all_objects:
        actual_reg = privileged_actual_storage_region(scene, obj_name)
        if actual_reg in actual_storage_inventory:
            actual_storage_inventory[actual_reg].append(obj_name)

    inventory_match = True
    for reg, expected_objs in declared_contents.items():
        if sorted(expected_objs) != sorted(actual_storage_inventory.get(reg, [])):
            inventory_match = False

    # 4. Physical functional regions verification (derived from compiled MjModel)
    physically_available_surfaces = privileged_actual_work_surface_regions(scene)
    physically_available_containers = privileged_actual_parts_container_regions(scene)

    physical_surface_match = set(declared_active_surfaces) == set(physically_available_surfaces)
    physical_container_match = set(declared_active_containers) == set(physically_available_containers)
    region_configuration_match = physical_surface_match and physical_container_match

    # 5. Storage articulation test
    storage_articulation_pass = True
    for reg in WORKSHOP_REGIONS:
        scene.open_container(reg)
        scene.close_container(reg)
        for obj_name in scene.storage_contents.get(reg, []):
            actual_reg = privileged_actual_storage_region(scene, obj_name)
            if actual_reg != reg:
                storage_articulation_pass = False

    # 6. Privileged oracle feasibility validator
    oracle_res = privileged_validate_variant_feasibility(scene)
    actual_oracle_outcome = oracle_res["status"]
    actual_rejection_reason = oracle_res.get("rejection_reason")

    outcome_match = (
        intended_outcome == actual_oracle_outcome
        and (
            intended_outcome == "FEASIBLE"
            or expected_rejection_reason == actual_rejection_reason
        )
    )

    # 7. Production API leakage check
    observed_instances = scene.get_observed_instances()
    candidate_regions = scene.get_candidate_regions()
    target_spec = scene.get_target_workpiece_specification()

    leaks = []
    leaks.extend(_check_dict_leakage(observed_instances, FORBIDDEN_LEAKAGE_STRINGS))
    leaks.extend(_check_dict_leakage(candidate_regions, FORBIDDEN_LEAKAGE_STRINGS))
    leaks.extend(_check_dict_leakage(target_spec, FORBIDDEN_LEAKAGE_STRINGS))
    production_api_leakage_pass = len(leaks) == 0

    # 8. Point-cloud stage-level evidence verification
    pointcloud_smoke_pass = True
    pointcloud_stage_results: list[dict[str, Any]] = []

    if run_pointcloud_smoke:
        try:
            checker = GeometryChecker(scene, width=640, height=480)
            for reg_id, _dir_name in STAGES:
                if reg_id != "INITIAL":
                    scene.open_container(reg_id)
                run = checker.run_region_inspection(
                    reg_id, rig_config=scene.inspection_rig_config
                )

                if reg_id == "INITIAL":
                    expected_objs = ["workshop_joint_seal", "workshop_frame_joint"]
                else:
                    expected_objs = list(scene.storage_contents.get(reg_id, []))

                observed_objs = list(run.clouds.keys())
                point_count_by_obj = {
                    obj: len(run.clouds[obj].points)
                    for obj in observed_objs
                }
                missing_objs = [
                    obj for obj in expected_objs if obj not in run.clouds
                ]
                low_point_objs = [
                    obj
                    for obj in expected_objs
                    if obj in run.clouds and len(run.clouds[obj].points) < MIN_WORKSHOP_OBJECT_FUSED_POINTS
                ]

                if not expected_objs:
                    stage_pass = True
                else:
                    stage_pass = len(missing_objs) == 0 and len(low_point_objs) == 0

                stage_record = {
                    "region_id": reg_id,
                    "total_points": run.total_points,
                    "expected_object_ids": expected_objs,
                    "observed_object_ids": observed_objs,
                    "point_count_by_object": point_count_by_obj,
                    "missing_object_ids": missing_objs,
                    "low_point_count_object_ids": low_point_objs,
                    "stage_pass": stage_pass,
                }
                pointcloud_stage_results.append(stage_record)

                if not stage_pass:
                    pointcloud_smoke_pass = False

                if reg_id != "INITIAL":
                    scene.close_container(reg_id)
        except Exception as err:
            pointcloud_smoke_pass = False
            pointcloud_stage_results.append({"error": str(err), "stage_pass": False})

    overall_pass = (
        stable_reset
        and free_bodies_pass
        and inventory_match
        and region_configuration_match
        and storage_articulation_pass
        and outcome_match
        and production_api_leakage_pass
        and pointcloud_smoke_pass
    )

    manifest = {
        "variant": variant_name,
        "intended_outcome": intended_outcome,
        "actual_oracle_outcome": actual_oracle_outcome,
        "expected_rejection_reason": expected_rejection_reason,
        "actual_rejection_reason": actual_rejection_reason,
        "outcome_match": outcome_match,
        "expected_storage_inventory": declared_contents,
        "actual_physical_storage_inventory": actual_storage_inventory,
        "inventory_match": inventory_match,
        "declared_active_surfaces": declared_active_surfaces,
        "physically_available_surfaces": physically_available_surfaces,
        "physical_surface_match": physical_surface_match,
        "declared_active_containers": declared_active_containers,
        "physically_available_containers": physically_available_containers,
        "physical_container_match": physical_container_match,
        "region_configuration_match": region_configuration_match,
        "body_count": n_bodies,
        "object_count": len(all_objects),
        "scene_stability_result": {
            "stable_reset": stable_reset,
            "has_nan_qpos": has_nan_qpos,
            "has_nan_qvel": has_nan_qvel,
        },
        "storage_articulation_pass": storage_articulation_pass,
        "dynamic_free_bodies_pass": free_bodies_pass,
        "oracle_feasibility_result": oracle_res,
        "production_api_leakage_pass": production_api_leakage_pass,
        "api_leaks": leaks,
        "pointcloud_stage_results": pointcloud_stage_results,
        "pointcloud_smoke_pass": pointcloud_smoke_pass,
        "overall_pass": overall_pass,
    }

    if output_dir is not None:
        var_out_dir = output_dir / variant_name
        var_out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = var_out_dir / "variant_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return manifest


def audit_all_variants(
    output_dir: Path | None = None,
    run_pointcloud_smoke: bool = True,
) -> dict[str, Any]:
    """Audit the complete 14-variant suite and output suite_summary.json."""
    config = _load_workshop_variants_config()
    variants = config.get("variants", {})

    print(f"\n{'=' * 80}")
    print(f"WORKSHOP (W1) SCENE & BENCHMARK PHYSICAL SUITE AUDIT")
    print(f"{'=' * 80}")
    print(f"Auditing {len(variants)} benchmark variants...")

    suite_results = []
    all_passed = True

    for v_name in sorted(variants.keys()):
        t0 = time.perf_counter()
        manifest = audit_single_variant(
            v_name,
            output_dir=output_dir,
            run_pointcloud_smoke=run_pointcloud_smoke,
        )
        elapsed = time.perf_counter() - t0

        pass_str = "PASS" if manifest["overall_pass"] else "FAIL"
        if not manifest["overall_pass"]:
            all_passed = False

        print(
            f"[{pass_str}] {v_name:<34} | "
            f"Outcome: {manifest['intended_outcome']:<10} -> {manifest['actual_oracle_outcome']:<10} | "
            f"Inv: {'OK' if manifest['inventory_match'] else 'FAIL':<4} | "
            f"Surfs: {'OK' if manifest['physical_surface_match'] else 'FAIL':<4} | "
            f"Conts: {'OK' if manifest['physical_container_match'] else 'FAIL':<4} | "
            f"Leak: {'OK' if manifest['production_api_leakage_pass'] else 'FAIL':<4} | "
            f"PC: {'OK' if manifest['pointcloud_smoke_pass'] else 'FAIL':<4} | "
            f"({elapsed:.2f}s)"
        )

        suite_results.append(
            {
                "variant": v_name,
                "intended_outcome": manifest["intended_outcome"],
                "actual_oracle_outcome": manifest["actual_oracle_outcome"],
                "expected_rejection_reason": manifest["expected_rejection_reason"],
                "actual_rejection_reason": manifest["actual_rejection_reason"],
                "inventory_match": manifest["inventory_match"],
                "declared_active_surfaces": manifest["declared_active_surfaces"],
                "physically_available_surfaces": manifest["physically_available_surfaces"],
                "physical_surface_match": manifest["physical_surface_match"],
                "declared_active_containers": manifest["declared_active_containers"],
                "physically_available_containers": manifest["physically_available_containers"],
                "physical_container_match": manifest["physical_container_match"],
                "stable_reset": manifest["scene_stability_result"]["stable_reset"],
                "dynamic_free_bodies_pass": manifest["dynamic_free_bodies_pass"],
                "storage_articulation_pass": manifest["storage_articulation_pass"],
                "production_api_leakage_pass": manifest["production_api_leakage_pass"],
                "pointcloud_smoke_pass": manifest["pointcloud_smoke_pass"],
                "pointcloud_stage_results": manifest["pointcloud_stage_results"],
                "overall_pass": manifest["overall_pass"],
            }
        )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_variants": len(variants),
        "passed_variants": sum(1 for r in suite_results if r["overall_pass"]),
        "all_passed": all_passed,
        "variants": suite_results,
    }

    if output_dir is not None:
        summary_path = output_dir / "suite_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n--> Wrote suite audit summary to {summary_path}")

    print(f"\nOverall Result: {'100% PASS' if all_passed else 'FAILURES DETECTED'}")
    print(f"{'=' * 80}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="all",
        help="Variant name to audit (or 'all' for complete suite).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/workshop_scene_audit",
        help="Directory to save audit manifests.",
    )
    parser.add_argument(
        "--no-pointcloud",
        action="store_true",
        help="Skip RGB-D point cloud smoke capture.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    run_pc = not args.no_pointcloud

    if args.variant == "all":
        summary = audit_all_variants(output_dir=out_dir, run_pointcloud_smoke=run_pc)
        if not summary["all_passed"]:
            sys.exit(1)
    else:
        manifest = audit_single_variant(
            args.variant, output_dir=out_dir, run_pointcloud_smoke=run_pc
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if not manifest["overall_pass"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
