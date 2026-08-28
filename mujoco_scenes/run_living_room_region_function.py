"""CLI for integrated one-shot living-room region-function Phase 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from mujoco_scenes.living_room_region_function import (
    DEFAULT_SEMANTIC_VOCABULARY,
    DEFAULT_TASK_CONFIG,
    IntegratedLivingRoomRegionRun,
    write_resolved_integrated_rig,
)
from mujoco_scenes.living_room_region_scene import (
    L2_INTEGRATED_SCENES,
    L2LivingRoomRegionScene,
)
from mujoco_scenes.region_ablation import create_region_semantic_detector
from mujoco_scenes.region_ablation2 import DEFAULT_EVALUATION_CONFIG
from mujoco_scenes.semantic_grounding import NullSemanticDetector, load_semantic_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ground and globally allocate living-room REGION functions."
    )
    parser.add_argument("--scene", choices=L2_INTEGRATED_SCENES, default=L2_INTEGRATED_SCENES[0])
    parser.add_argument("--no-robot", action="store_true", default=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--semantic-vocabulary", type=Path, default=DEFAULT_SEMANTIC_VOCABULARY)
    parser.add_argument("--semantic-detector", choices=("yolo_world", "none"), default="yolo_world")
    parser.add_argument("--semantic-model", default="semantic_model_cache/yolov8m-worldv2.pt")
    parser.add_argument("--semantic-confidence-threshold", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--skip-expectation-validation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = (args.runs_root / args.run_id).resolve()
    rig_path = run_dir.parent / f".{args.run_id}_resolved_rig.yaml"
    write_resolved_integrated_rig(args.scene, rig_path)
    if args.semantic_detector == "yolo_world":
        detector, semantic_config = create_region_semantic_detector(
            checkpoint=args.semantic_model,
            confidence_threshold=args.semantic_confidence_threshold,
            vocabulary_path=args.semantic_vocabulary,
        )
    else:
        detector = NullSemanticDetector()
        semantic_config = load_semantic_config(vocabulary_path=args.semantic_vocabulary)
    scene = L2LivingRoomRegionScene(args.scene, robot="none")
    run = IntegratedLivingRoomRegionRun(
        run_dir,
        scene_name=args.scene,
        task_config=args.task_config,
        evaluation_config=DEFAULT_EVALUATION_CONFIG,
        rig_config=rig_path,
        semantic_detector=detector,
        semantic_config=semantic_config,
        width=args.width,
        height=args.height,
    ).run(scene)
    validation = run.validate_expected()
    print(
        f"[L2 REGION FUNCTION] joint/global: {run.production_result['status']}"
    )
    print(f"[L2 REGION FUNCTION] output: {run.run_dir}")
    print(f"[L2 REGION FUNCTION] expected outcome: {validation['passed']}")
    if not args.skip_expectation_validation and not validation["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
