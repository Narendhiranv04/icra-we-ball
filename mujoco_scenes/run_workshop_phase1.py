"""CLI runner for Workshop Phase 1 Grounding and Joint Requirement Satisfaction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from mujoco_scenes.workshop_phase1.evaluation import PrivilegedPhase1Evaluator
from mujoco_scenes.workshop_phase1.inspection_controller import (
    WorkshopPhase1InspectionController,
)
from mujoco_scenes.workshop_phase1.types import (
    AblationType,
    MaskBackendType,
    ProposalMode,
    SemanticBackendType,
)
from mujoco_scenes.workshop_scene import WorkshopScene

ALL_WORKSHOP_VARIANTS = [
    "F0_BASE",
    "F1_TOOL_ALTERNATIVE",
    "F2_REGION_ALTERNATIVE",
    "F3_DISTRIBUTED_OBJECTS",
    "F4_OBJECT_REGION_COUPLING",
    "F5_DECOY_HEAVY",
    "F6_LAYOUT_SWAPPED",
    "I0_NO_VALID_DRIVER",
    "I1_NO_VALID_FASTENER",
    "I2_NO_WORK_SURFACE",
    "I3_NO_PARTS_CONTAINER",
    "I4_TOOL_GEOMETRY_FAILURE",
    "I5_OBJECT_REGION_PACKING_FAILURE",
    "I6_GLOBAL_CONFLICT",
]


def parse_mask_backend(name: str) -> MaskBackendType:
    if name == "oracle":
        return MaskBackendType.ORACLE
    elif name == "connected_component":
        return MaskBackendType.CONNECTED_COMPONENT
    return MaskBackendType.PRODUCTION


def parse_semantic_backend(name: str) -> SemanticBackendType:
    if name == "oracle":
        return SemanticBackendType.ORACLE
    elif name == "deterministic_test":
        return SemanticBackendType.DETERMINISTIC_TEST
    return SemanticBackendType.PRODUCTION


def parse_ablation(name: str) -> AblationType:
    mapping = {
        "none": AblationType.NONE,
        "semantic_only": AblationType.SEMANTIC_ONLY,
        "no_geometry": AblationType.NO_GEOMETRY,
        "no_joint_coupling": AblationType.NO_JOINT_COUPLING,
        "no_persistence": AblationType.NO_PERSISTENCE,
        "single_view": AblationType.SINGLE_FRONT_VIEW,
        "single_front_view": AblationType.SINGLE_FRONT_VIEW,
        "oracle_mask": AblationType.ORACLE_MASK,
        "oracle_semantics": AblationType.ORACLE_SEMANTICS,
    }
    return mapping.get(name.lower(), AblationType.NONE)


def parse_proposal_mode(name: str) -> ProposalMode:
    if name == "yolo_plus_rgbd_fallback":
        return ProposalMode.YOLO_PLUS_RGBD_FALLBACK
    return ProposalMode.YOLO_ONLY


def run_single_variant(
    variant: str,
    robot: str = "none",
    mask_backend: str = "production",
    semantic_backend: str = "production",
    proposal_mode: str = "yolo_only",
    ablation: str = "none",
    output_dir: Path | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run Phase 1 pipeline on one variant."""
    print(f"\n==================================================")
    print(f"Running Workshop Phase 1: {variant}")
    print(f"  Mask: {mask_backend}, Semantics: {semantic_backend}, Proposal: {proposal_mode}, Ablation: {ablation}")
    print(f"==================================================")

    scene = WorkshopScene(robot=robot, variant=variant)

    m_type = parse_mask_backend(mask_backend)
    s_type = parse_semantic_backend(semantic_backend)
    abl_type = parse_ablation(ablation)
    prop_mode = parse_proposal_mode(proposal_mode)

    var_out = (output_dir / variant) if output_dir else None

    controller = WorkshopPhase1InspectionController(
        scene=scene,
        mask_backend=m_type,
        semantic_backend=s_type,
        ablation=abl_type,
        proposal_mode=prop_mode,
        output_dir=var_out,
    )

    result = controller.run_episode()

    eval_metrics = {}
    if evaluate:
        evaluator = PrivilegedPhase1Evaluator(scene)
        eval_metrics = evaluator.evaluate_episode(
            result=result,
            tracks=list(controller.tracker.tracks.values()),
            regions=controller.candidate_regions,
            output_dir=var_out,
        )

    print(f"Outcome: {result.status}")
    if result.witness:
        print(f"  Witness: driver={result.witness.driver_id}, fastener={result.witness.fastener_id}, surface={result.witness.work_surface_id}, container={result.witness.parts_container_id}")
    if result.rejection_reason:
        print(f"  Rejection Reason: {result.rejection_reason}")
    print(f"  Inspected Regions: {result.trace.inspected_regions} (Early stopped: {result.trace.early_stopped})")
    if evaluate:
        print(f"  Evaluation Pass: {eval_metrics.get('overall_pass', False)}")

    return {
        "variant": variant,
        "result": result.to_dict(),
        "eval_metrics": eval_metrics,
    }


def run_benchmark_suite(
    variants: list[str],
    robot: str = "none",
    mask_backend: str = "production",
    semantic_backend: str = "production",
    proposal_mode: str = "yolo_only",
    ablation: str = "none",
    output_dir: Path | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run all specified variants and compile benchmark summary table."""
    results = []
    passed_count = 0

    print("\n" + "=" * 80)
    print(f"WORKSHOP (W1) PHASE 1 BENCHMARK SUITE ({len(variants)} variants)")
    print(f"Mask: {mask_backend} | Semantics: {semantic_backend} | Proposal: {proposal_mode} | Ablation: {ablation}")
    print("=" * 80)

    for var in variants:
        res = run_single_variant(
            variant=var,
            robot=robot,
            mask_backend=mask_backend,
            semantic_backend=semantic_backend,
            proposal_mode=proposal_mode,
            ablation=ablation,
            output_dir=output_dir,
            evaluate=evaluate,
        )
        results.append(res)
        if res["eval_metrics"].get("overall_pass", False):
            passed_count += 1

    summary = {
        "total_variants": len(variants),
        "passed_variants": passed_count,
        "accuracy": round(passed_count / max(1, len(variants)), 4),
        "mask_backend": mask_backend,
        "semantic_backend": semantic_backend,
        "proposal_mode": proposal_mode,
        "ablation": ablation,
        "results": results,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY: {passed_count} / {len(variants)} PASSED ({summary['accuracy']*100:.1f}%)")
    print("=" * 80)
    for res in results:
        v = res["variant"]
        status = res["result"]["status"]
        rej = res["result"]["rejection_reason"] or "N/A"
        epass = res["eval_metrics"].get("overall_pass", False)
        pass_str = "[PASS]" if epass else "[FAIL]"
        print(f"{pass_str} {v:<35} | Pred: {status:<10} | Rej: {rej:<30}")

    return summary


def run_ablation_suite(output_dir: Path | None = None) -> list[dict[str, Any]]:
    """Run full canonical 7-arm ablation suite across all 14 variants (98 total episodes)."""
    ablations = [
        {"name": "Full Production Pipeline", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "none"},
        {"name": "Oracle Masks (Upper Bound)", "mask": "oracle", "sem": "production", "prop": "yolo_only", "abl": "oracle_mask"},
        {"name": "Oracle Semantics", "mask": "production", "sem": "oracle", "prop": "yolo_only", "abl": "oracle_semantics"},
        {"name": "Semantic-Only Grounding (No Geo)", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "semantic_only"},
        {"name": "No Joint Coupling Checks", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "no_joint_coupling"},
        {"name": "No Multi-Stage Persistence", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "no_persistence"},
        {"name": "Single Front Camera View", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "single_front_view"},
    ]

    out_base = output_dir or Path("outputs/workshop_phase1_final/ablations")
    out_base.mkdir(parents=True, exist_ok=True)

    suite_results = []
    for abl in ablations:
        print(f"\n>>>>>>>> RUNNING ABLATION: {abl['name']} <<<<<<<<")
        abl_out = out_base / abl["abl"]
        summary = run_benchmark_suite(
            variants=ALL_WORKSHOP_VARIANTS,
            mask_backend=abl["mask"],
            semantic_backend=abl["sem"],
            proposal_mode=abl["prop"],
            ablation=abl["abl"],
            output_dir=abl_out,
            evaluate=True,
        )
        suite_results.append({
            "ablation_name": abl["name"],
            "mask_backend": abl["mask"],
            "semantic_backend": abl["sem"],
            "ablation_flag": abl["abl"],
            "passed": summary["passed_variants"],
            "total": summary["total_variants"],
            "accuracy": summary["accuracy"],
        })

    with open(out_base / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)

    with open(out_base / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation_name", "mask_backend", "semantic_backend", "ablation_flag", "passed", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(suite_results)

    print("\n" + "=" * 90)
    print("ALL ABLATIONS COMPLETE. SUMMARY TABLE:")
    print("=" * 90)
    for row in suite_results:
        print(f"{row['ablation_name']:<40} | {row['passed']:>2}/{row['total']} | Accuracy: {row['accuracy']*100:.1f}%")

    return suite_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Workshop Phase 1 Grounding Runner")
    parser.add_argument("--variant", type=str, default="F0_BASE", help="Variant name (e.g. F0_BASE or 'all')")
    parser.add_argument("--robot", type=str, default="none", help="Robot type ('none' or 'google_robot')")
    parser.add_argument("--mask-backend", type=str, default="production", choices=["production", "oracle", "connected_component"], help="Mask proposal backend")
    parser.add_argument("--semantic-backend", type=str, default="production", choices=["production", "oracle", "deterministic_test"], help="Semantic grounding backend")
    parser.add_argument("--proposal-mode", type=str, default="yolo_only", choices=["yolo_only", "yolo_plus_rgbd_fallback"], help="Proposal mode")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "semantic_only", "no_geometry", "no_joint_coupling", "no_persistence", "single_view", "single_front_view", "oracle_mask", "oracle_semantics", "run_suite"], help="Ablation mode")
    parser.add_argument("--output", type=str, default="outputs/workshop_phase1_final", help="Output directory")
    parser.add_argument("--evaluate", action="store_true", default=True, help="Compute privileged post-hoc metrics")
    parser.add_argument("--list-variants", action="store_true", help="List all variants")

    args = parser.parse_args()

    if args.list_variants:
        print("Available Workshop Variants:")
        for v in ALL_WORKSHOP_VARIANTS:
            print(f"  - {v}")
        return

    out_dir = Path(args.output) if args.output else None

    if args.ablation == "run_suite":
        run_ablation_suite(output_dir=out_dir)
    elif args.variant == "all":
        run_benchmark_suite(
            variants=ALL_WORKSHOP_VARIANTS,
            robot=args.robot,
            mask_backend=args.mask_backend,
            semantic_backend=args.semantic_backend,
            proposal_mode=args.proposal_mode,
            ablation=args.ablation,
            output_dir=out_dir,
            evaluate=args.evaluate,
        )
    else:
        run_single_variant(
            variant=args.variant,
            robot=args.robot,
            mask_backend=args.mask_backend,
            semantic_backend=args.semantic_backend,
            proposal_mode=args.proposal_mode,
            ablation=args.ablation,
            output_dir=out_dir,
            evaluate=args.evaluate,
        )


if __name__ == "__main__":
    main()
