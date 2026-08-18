"""CLI runner for Workshop Phase 1 Grounding and Joint Requirement Satisfaction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from mujoco_scenes.workshop_phase1.evaluation import PrivilegedPhase1Evaluator
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter
from mujoco_scenes.workshop_phase1.inspection_controller import WorkshopPhase1InspectionController
from mujoco_scenes.workshop_phase1.requirements import (
    FMRequirementProvider,
    StaticWorkshopRequirementProvider,
)
from mujoco_scenes.workshop_phase1.types import MaskBackendType
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


def run_single_variant(
    variant: str,
    robot: str = "none",
    mask_backend: str = "production",
    requirements_source: str = "static",
    output_dir: Path | None = None,
    evaluate: bool = False,
) -> dict[str, Any]:
    """Run Phase 1 pipeline on one variant."""
    print(f"\n==================================================")
    print(f"Running Workshop Phase 1: {variant}")
    print(f"  Robot: {robot}, Mask Backend: {mask_backend}, Reqs: {requirements_source}")
    print(f"==================================================")

    # Initialize scene
    scene = WorkshopScene(robot=robot, variant=variant)

    # Configure mask backend
    if mask_backend == "oracle":
        m_type = MaskBackendType.ORACLE
    elif mask_backend == "connected_component":
        m_type = MaskBackendType.CONNECTED_COMPONENT
    else:
        m_type = MaskBackendType.PRODUCTION

    # Configure requirements
    if requirements_source == "fm":
        req_provider = FMRequirementProvider()
    else:
        req_provider = StaticWorkshopRequirementProvider()

    var_out = (output_dir / variant) if output_dir else None

    controller = WorkshopPhase1InspectionController(
        mask_backend=m_type,
        requirements_provider=req_provider,
        output_dir=var_out,
    )

    result = controller.run_episode(scene)

    eval_metrics = {}
    if evaluate:
        evaluator = PrivilegedPhase1Evaluator(scene)
        eval_metrics = evaluator.evaluate_episode(
            result=result,
            tracks=list(controller.tracker.tracks.values()),
            regions=controller.region_grounder._known_regions.values(),
            output_dir=var_out,
        )

    print(f"Outcome: {result.status}")
    if result.witness:
        print(f"  Witness: driver={result.witness.driver_id}, fastener={result.witness.fastener_id}, surface={result.witness.work_surface_id}, container={result.witness.parts_container_id}")
    if result.rejection_reason:
        print(f"  Rejection Reason: {result.rejection_reason}")
    print(f"  Inspected Regions: {result.trace.inspected_regions} (Early stopped: {result.trace.early_stopped})")
    print(f"  Time: {result.metrics.get('total_inference_time_s', 0):.2f}s")
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
    requirements_source: str = "static",
    output_dir: Path | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run all specified variants and compile benchmark summary table."""
    results = []
    passed_count = 0

    print("\n" + "=" * 80)
    print(f"WORKSHOP (W1) PHASE 1 BENCHMARK SUITE ({len(variants)} variants)")
    print(f"Mask Backend: {mask_backend} | Reqs: {requirements_source}")
    print("=" * 80)

    for var in variants:
        res = run_single_variant(
            variant=var,
            robot=robot,
            mask_backend=mask_backend,
            requirements_source=requirements_source,
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
        "requirements_source": requirements_source,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Workshop Phase 1 Grounding Runner")
    parser.add_argument("--variant", type=str, default="F0_BASE", help="Variant name (e.g. F0_BASE or 'all')")
    parser.add_argument("--robot", type=str, default="none", help="Robot type ('none' or 'google_robot')")
    parser.add_argument("--mask-backend", type=str, default="production", choices=["production", "oracle", "connected_component"], help="Mask proposal backend")
    parser.add_argument("--requirements-source", type=str, default="static", choices=["static", "fm"], help="Requirement provider")
    parser.add_argument("--inspection-policy", type=str, default="fixed", choices=["fixed", "fm"], help="Inspection policy")
    parser.add_argument("--output", type=str, default="outputs/workshop_phase1", help="Output directory")
    parser.add_argument("--evaluate", action="store_true", default=True, help="Compute privileged post-hoc metrics")
    parser.add_argument("--list-variants", action="store_true", help="List all variants")

    args = parser.parse_args()

    if args.list_variants:
        print("Available Workshop Variants:")
        for v in ALL_WORKSHOP_VARIANTS:
            print(f"  - {v}")
        return

    out_dir = Path(args.output) if args.output else None

    if args.variant == "all":
        run_benchmark_suite(
            variants=ALL_WORKSHOP_VARIANTS,
            robot=args.robot,
            mask_backend=args.mask_backend,
            requirements_source=args.requirements_source,
            output_dir=out_dir,
            evaluate=args.evaluate,
        )
    else:
        run_single_variant(
            variant=args.variant,
            robot=args.robot,
            mask_backend=args.mask_backend,
            requirements_source=args.requirements_source,
            output_dir=out_dir,
            evaluate=args.evaluate,
        )


if __name__ == "__main__":
    main()
