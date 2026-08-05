"""Compare image-only segmentation against the MuJoCo oracle baseline."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from mujoco_scenes.geometry_checker import GeometryChecker, PointCloudRun
from mujoco_scenes.scene_loader import CONTAINER_JOINTS, KitchenScene


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def compare_runs(
    oracle: PointCloudRun,
    learned: PointCloudRun,
    *,
    match_iou: float = 0.5,
) -> dict:
    if oracle.inspection is None or learned.inspection is None:
        raise ValueError("comparison requires region-inspection runs")
    cameras = {}
    total_oracle = total_learned = total_matches = 0
    matched_ious: list[float] = []
    for camera_id, oracle_capture in oracle.inspection.cameras.items():
        learned_capture = learned.inspection.cameras[camera_id]
        pairs = sorted(
            (
                (mask_iou(oracle_mask, learned_mask), oracle_id, learned_id)
                for oracle_id, oracle_mask in oracle_capture.instance_masks.items()
                for learned_id, learned_mask in learned_capture.instance_masks.items()
            ),
            reverse=True,
        )
        used_oracle: set[str] = set()
        used_learned: set[str] = set()
        matches = []
        for iou, oracle_id, learned_id in pairs:
            if iou < match_iou:
                break
            if oracle_id in used_oracle or learned_id in used_learned:
                continue
            used_oracle.add(oracle_id)
            used_learned.add(learned_id)
            matches.append(float(iou))
        oracle_count = len(oracle_capture.instance_masks)
        learned_count = len(learned_capture.instance_masks)
        cameras[camera_id] = {
            "oracle_masks": oracle_count,
            "learned_masks": learned_count,
            "matched_masks": len(matches),
            "matched_mean_iou": float(np.mean(matches)) if matches else None,
        }
        total_oracle += oracle_count
        total_learned += learned_count
        total_matches += len(matches)
        matched_ious.extend(matches)
    return {
        "match_iou_threshold": match_iou,
        "aggregate": {
            "oracle_masks": total_oracle,
            "learned_masks": total_learned,
            "matched_masks": total_matches,
            "precision": total_matches / total_learned if total_learned else 0.0,
            "recall": total_matches / total_oracle if total_oracle else 0.0,
            "matched_mean_iou": (
                float(np.mean(matched_ious)) if matched_ious else None
            ),
        },
        "cameras": cameras,
    }


def _load_segmenter(spec: str):
    module_name, separator, factory_name = spec.partition(":")
    if not separator:
        raise ValueError("segmenter must use package.module:factory")
    return getattr(importlib.import_module(module_name), factory_name)()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="S1_coffee_missing_mug")
    parser.add_argument(
        "--region", choices=("INITIAL", *CONTAINER_JOINTS), default="INITIAL"
    )
    parser.add_argument(
        "--segmenter", default="mujoco_scenes.sam3_client:create_segmenter"
    )
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/sam3_comparison"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args(argv)

    scene = KitchenScene(args.scene, robot="google")
    if args.region != "INITIAL":
        scene.open_container(args.region, steps=500)
    oracle = GeometryChecker(scene, width=args.width, height=args.height)
    learned = GeometryChecker(
        scene,
        width=args.width,
        height=args.height,
        segmenter=_load_segmenter(args.segmenter),
        semantic_prompts=args.prompt,
    )
    oracle_run = oracle.run_region_inspection(
        args.region, stage_output_dir=args.output / "oracle"
    )
    learned_run = learned.run_region_inspection(
        args.region, stage_output_dir=args.output / "sam3"
    )
    report = compare_runs(oracle_run, learned_run, match_iou=args.match_iou)
    report["scene"] = args.scene
    report["region"] = args.region
    report["prompts"] = args.prompt
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Comparison: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
