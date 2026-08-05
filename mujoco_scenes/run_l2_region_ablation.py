"""Command-line runner for the L2 living-room region-grounding benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from mujoco_scenes.living_room_region_scene import (
    L2_SCENES,
    L2LivingRoomRegionScene,
)
from mujoco_scenes.region_ablation import (
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_RIG_CONFIG,
    DEFAULT_SEMANTIC_VOCABULARY,
    DEFAULT_TASK_CONFIG,
    RegionAblationRun,
    create_region_semantic_detector,
)
from mujoco_scenes.semantic_grounding import (
    NullSemanticDetector,
    load_semantic_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture L2 region evidence once and evaluate geometry-only, "
            "semantic-only, and joint grounding from that evidence."
        )
    )
    parser.add_argument("--scene", choices=L2_SCENES, default=L2_SCENES[0])
    parser.add_argument("--robot", choices=("google", "none"), default="none")
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Alias for --robot none.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=DEFAULT_EVALUATION_CONFIG,
    )
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--semantic-vocabulary",
        type=Path,
        default=DEFAULT_SEMANTIC_VOCABULARY,
    )
    parser.add_argument(
        "--semantic-detector",
        choices=("yolo_world", "none"),
        default="yolo_world",
    )
    parser.add_argument(
        "--semantic-model",
        default="semantic_model_cache/yolov8m-worldv2.pt",
    )
    parser.add_argument(
        "--semantic-confidence-threshold", type=float, default=0.03
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--full-order",
        action="store_true",
        help="Capture every ranked candidate even after joint completion.",
    )
    parser.add_argument(
        "--skip-expectation-validation",
        action="store_true",
        help="Do not fail when evaluation-only expected outcomes differ.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    robot = "none" if arguments.no_robot else arguments.robot
    if arguments.semantic_detector == "yolo_world":
        detector, semantic_config = create_region_semantic_detector(
            checkpoint=arguments.semantic_model,
            confidence_threshold=arguments.semantic_confidence_threshold,
            vocabulary_path=arguments.semantic_vocabulary,
        )
    else:
        detector = NullSemanticDetector()
        semantic_config = load_semantic_config(
            vocabulary_path=arguments.semantic_vocabulary
        )
    scene = L2LivingRoomRegionScene(arguments.scene, robot=robot)
    run_dir = arguments.runs_root / arguments.run_id
    run = RegionAblationRun(
        run_dir,
        scene_name=scene.scene_name,
        task_config=arguments.task_config,
        evaluation_config=arguments.evaluation_config,
        rig_config=arguments.rig_config,
        semantic_detector=detector,
        semantic_config=semantic_config,
        width=arguments.width,
        height=arguments.height,
    )
    run.run(scene, full_order=arguments.full_order)
    summary = run.evaluate_same_evidence()
    validation = run.validate_expected_outcomes()
    print(f"[REGION] production status: {run.production_status}")
    for mode, result in summary["modes"].items():
        print(
            f"[REGION] {mode}: {result['status']} "
            f"at {result['selected_inspection_label']}"
        )
    print(f"[REGION] output: {run.run_dir}")
    print(f"[REGION] expected outcomes passed: {validation['passed']}")
    if not arguments.skip_expectation_validation and not validation["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
