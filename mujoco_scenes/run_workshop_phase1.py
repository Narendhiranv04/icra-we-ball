"""CLI runner for Workshop Phase 1 Grounding and Joint Requirement Satisfaction."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any
import yaml

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from mujoco_scenes.workshop_phase1.evaluation import PrivilegedPhase1Evaluator
from mujoco_scenes.workshop_phase1.inspection_controller import (
    WorkshopPhase1InspectionController,
)
from mujoco_scenes.workshop_phase1.requirements import ManualWorkshopFMContract
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
            detection_diagnostics=controller.detection_diagnostics,
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
    detail_variants: set[str] | None = None,
) -> dict[str, Any]:
    """Run all specified variants and compile benchmark summary table."""
    results = []
    passed_count = 0

    print("\n" + "=" * 80)
    print(f"WORKSHOP (W1) PHASE 1 BENCHMARK SUITE ({len(variants)} variants)")
    print(f"Mask: {mask_backend} | Semantics: {semantic_backend} | Proposal: {proposal_mode} | Ablation: {ablation}")
    print("=" * 80)

    for var in variants:
        variant_output = output_dir
        if detail_variants is not None and var not in detail_variants:
            variant_output = None
        res = run_single_variant(
            variant=var,
            robot=robot,
            mask_backend=mask_backend,
            semantic_backend=semantic_backend,
            proposal_mode=proposal_mode,
            ablation=ablation,
            output_dir=variant_output,
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


def run_ablation_suite(output_dir: Path | None = None, concise: bool = False,
                       controlled: bool = False) -> list[dict[str, Any]]:
    """Run separately labelled production or controlled-perception ablations."""
    if controlled:
        ablations = [
            {"name": "Controlled Full Grounding", "mask": "oracle", "sem": "oracle", "prop": "yolo_only", "abl": "none"},
            {"name": "Controlled Semantic Only", "mask": "oracle", "sem": "oracle", "prop": "yolo_only", "abl": "semantic_only"},
            {"name": "Controlled No Geometry", "mask": "oracle", "sem": "oracle", "prop": "yolo_only", "abl": "no_geometry"},
            {"name": "Controlled No Joint Coupling", "mask": "oracle", "sem": "oracle", "prop": "yolo_only", "abl": "no_joint_coupling"},
            {"name": "Controlled No Persistence", "mask": "oracle", "sem": "oracle", "prop": "yolo_only", "abl": "no_persistence"},
            {"name": "Controlled Single Front View", "mask": "oracle", "sem": "oracle", "prop": "yolo_only", "abl": "single_front_view"},
        ]
    else:
        ablations = [
            {"name": "Full Production Pipeline", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "none"},
            {"name": "Production Semantic Only", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "semantic_only"},
            {"name": "Production No Joint Coupling", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "no_joint_coupling"},
            {"name": "Production No Persistence", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "no_persistence"},
            {"name": "Production Single Front View", "mask": "production", "sem": "production", "prop": "yolo_only", "abl": "single_front_view"},
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
            detail_variants=set() if concise else None,
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


def aggregate_semantic_diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    additive = ("visible_gt_count", "yolo_detection_count", "matched_gt_count",
                "false_positive_count", "duplicate_detections",
                "association_correct_count", "association_evaluated_count")
    for result in summary["results"]:
        rows = result.get("eval_metrics", {}).get("semantic_diagnostics", {}).get("categories", {})
        for category, row in rows.items():
            target = categories.setdefault(category, {key: 0 for key in additive})
            for key in additive:
                target[key] += int(row.get(key, 0))
    for row in categories.values():
        row["recall"] = round(row["matched_gt_count"] / max(1, row["visible_gt_count"]), 4)
        row["association_accuracy"] = round(
            row["association_correct_count"] / max(1, row["association_evaluated_count"]), 4)
    return {"categories": categories, "privileged_use": "POST_HOC_EVALUATION_ONLY"}


def production_failure_breakdown(summary: dict[str, Any]) -> dict[str, Any]:
    failures = []
    for row in summary["results"]:
        if row["eval_metrics"].get("overall_pass"):
            continue
        result = row["result"]
        diagnostics = row["eval_metrics"].get("semantic_diagnostics", {}).get("categories", {})
        if result["status"] == "INSUFFICIENT_EVIDENCE":
            layer = "INSUFFICIENT_EVIDENCE"
        elif result["rejection_reason"] in {"NO_WORK_SURFACE", "NO_PARTS_CONTAINER"}:
            region_detections = sum(diagnostics.get(name, {}).get("yolo_detection_count", 0)
                                    for name in ("workbench", "tool_cart", "shelf", "parts_tray", "hardware_bin"))
            layer = "REGION_ASSOCIATION_FAILURE" if region_detections else "DETECTION_MISS"
        elif row["eval_metrics"].get("tracked_objects_count", 0) == 0:
            layer = "DETECTION_MISS"
        elif result["rejection_reason"] in {"TOOL_GEOMETRY_FAILURE", "NO_VALID_DRIVER", "NO_VALID_FASTENER"}:
            layer = "SEMANTIC_MISCLASSIFICATION"
        elif result["rejection_reason"] == "OBJECT_REGION_PACKING_FAILURE":
            layer = "JOINT_RELATION_FALSE"
        elif result["rejection_reason"] == "GLOBAL_CONFLICT":
            layer = "JOINT_RELATION_FALSE"
        else:
            layer = "OTHER"
        failures.append({
            "variant": row["variant"], "layer": layer,
            "predicted_status": result["status"],
            "predicted_rejection": result["rejection_reason"],
            "expected_status": row["eval_metrics"].get("expected_status"),
            "expected_rejection": row["eval_metrics"].get("expected_rejection"),
        })
    return {"classification_basis": "Post-hoc evaluator diagnostics; never used by production decisions.",
            "failures": failures}


def run_canonical_suite(output_dir: Path) -> dict[str, Any]:
    """Reproduce all canonical Phase-1 summaries without a scratch script."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_root = WORKSPACE_ROOT / "mujoco_scenes" / "configs"
    shutil.copyfile(config_root / "workshop_phase1_fm_contract.yaml",
                    output_dir / "resolved_fm_contract.yaml")
    shutil.copyfile(config_root / "workshop_geometry_inference.yaml",
                    output_dir / "resolved_geometry_config.yaml")
    runtime = yaml.safe_load((config_root / "workshop_phase1.yaml").read_text())
    (output_dir / "resolved_runtime_config.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8")
    contract = ManualWorkshopFMContract(config_root / "workshop_phase1_fm_contract.yaml")
    (output_dir / "yolo_vocabulary.json").write_text(json.dumps({
        "ranked_detector_vocabulary": contract.get_ranked_detector_vocabulary(),
        "detector_classes": contract.get_detector_prompts(),
        "detector_label_to_canonical": contract.get_detector_label_to_canonical_map(),
        "aliases_are_detector_classes": False,
    }, indent=2), encoding="utf-8")
    representative = {"F4_OBJECT_REGION_COUPLING", "I1_NO_VALID_FASTENER",
                      "I4_TOOL_GEOMETRY_FAILURE", "I5_OBJECT_REGION_PACKING_FAILURE",
                      "I6_GLOBAL_CONFLICT"}
    oracle = run_benchmark_suite(
        ALL_WORKSHOP_VARIANTS, mask_backend="oracle", semantic_backend="oracle",
        ablation="oracle_semantics", output_dir=output_dir / "oracle_upper_bound",
        detail_variants=representative)
    oracle_masks = run_benchmark_suite(
        ALL_WORKSHOP_VARIANTS, mask_backend="oracle", semantic_backend="production",
        ablation="oracle_mask", output_dir=output_dir / "yolo_oracle_masks",
        detail_variants=set())
    production = run_benchmark_suite(
        ALL_WORKSHOP_VARIANTS, mask_backend="production", semantic_backend="production",
        ablation="none", output_dir=output_dir / "production", detail_variants=set())
    production_diagnostics = aggregate_semantic_diagnostics(production)
    (output_dir / "production" / "semantic_diagnostics_summary.json").write_text(
        json.dumps(production_diagnostics, indent=2), encoding="utf-8")
    (output_dir / "production" / "failure_breakdown.json").write_text(
        json.dumps(production_failure_breakdown(production), indent=2), encoding="utf-8")
    production_ablations = run_ablation_suite(
        output_dir / "production_ablations", concise=True, controlled=False)
    controlled_ablations = run_ablation_suite(
        output_dir / "controlled_ablations", concise=True, controlled=True)
    result = {"oracle_upper_bound": oracle["passed_variants"],
              "yolo_oracle_masks": oracle_masks["passed_variants"],
              "production": production["passed_variants"],
              "production_ablations": production_ablations,
              "controlled_ablations": controlled_ablations}
    (output_dir / "canonical_run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Workshop Phase 1 Grounding Runner")
    parser.add_argument("--variant", type=str, default="F0_BASE", help="Variant name (e.g. F0_BASE or 'all')")
    parser.add_argument("--robot", type=str, default="none", help="Robot type ('none' or 'google_robot')")
    parser.add_argument("--mask-backend", type=str, default="production", choices=["production", "oracle", "connected_component"], help="Mask proposal backend")
    parser.add_argument("--semantic-backend", type=str, default="production", choices=["production", "oracle", "deterministic_test"], help="Semantic grounding backend")
    parser.add_argument("--proposal-mode", type=str, default="yolo_only", choices=["yolo_only"], help="YOLO-World proposals with depth refinement")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "semantic_only", "no_geometry", "no_joint_coupling", "no_persistence", "single_view", "single_front_view", "oracle_mask", "oracle_semantics", "run_suite"], help="Ablation mode")
    parser.add_argument("--output", type=str, default="outputs/workshop_phase1_final", help="Output directory")
    parser.add_argument("--evaluate", action="store_true", default=True, help="Compute privileged post-hoc metrics")
    parser.add_argument("--list-variants", action="store_true", help="List all variants")
    parser.add_argument("--canonical", action="store_true", help="Run oracle, oracle-mask, production, and ablation canonical suites")

    args = parser.parse_args()

    if args.list_variants:
        print("Available Workshop Variants:")
        for v in ALL_WORKSHOP_VARIANTS:
            print(f"  - {v}")
        return

    out_dir = Path(args.output) if args.output else None

    if args.canonical:
        run_canonical_suite(out_dir or Path("outputs/workshop_phase1_final"))
    elif args.ablation == "run_suite":
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
