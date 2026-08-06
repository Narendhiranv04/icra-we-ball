"""CLI for living-room target-specific Region Ablation 3."""

from __future__ import annotations

import argparse
from pathlib import Path

from mujoco_scenes.living_room_region_scene import (
    L2_ABLATION3_SCENES,
    L2LivingRoomRegionScene,
)
from mujoco_scenes.region_ablation import create_region_semantic_detector
from mujoco_scenes.region_ablation3 import (
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_RIG_CONFIG,
    DEFAULT_SEMANTIC_VOCABULARY,
    DEFAULT_TASK_CONFIG,
    RegionAblation3Run,
)
from mujoco_scenes.semantic_grounding import (
    NullSemanticDetector,
    load_semantic_config,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--scene", choices=L2_ABLATION3_SCENES, required=True)
    result.add_argument("--robot", choices=("google", "none"), default="none")
    result.add_argument("--no-robot", action="store_true")
    result.add_argument("--runs-root", type=Path, default=Path("runs"))
    result.add_argument("--run-id", required=True)
    result.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    result.add_argument(
        "--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG
    )
    result.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    result.add_argument(
        "--semantic-vocabulary",
        type=Path,
        default=DEFAULT_SEMANTIC_VOCABULARY,
    )
    result.add_argument(
        "--semantic-detector",
        choices=("yolo_world", "none"),
        default="yolo_world",
    )
    result.add_argument(
        "--semantic-model",
        default="semantic_model_cache/yolov8m-worldv2.pt",
    )
    result.add_argument(
        "--semantic-confidence-threshold", type=float, default=0.03
    )
    result.add_argument("--width", type=int, default=1280)
    result.add_argument("--height", type=int, default=960)
    result.add_argument("--skip-expectation-validation", action="store_true")
    return result


def main() -> None:
    arguments = parser().parse_args()
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
    scene = L2LivingRoomRegionScene(
        arguments.scene,
        robot="none" if arguments.no_robot else arguments.robot,
    )
    run = RegionAblation3Run(
        arguments.runs_root / arguments.run_id,
        scene_name=scene.scene_name,
        task_config=arguments.task_config,
        evaluation_config=arguments.evaluation_config,
        rig_config=arguments.rig_config,
        semantic_detector=detector,
        semantic_config=semantic_config,
        width=arguments.width,
        height=arguments.height,
    ).run(scene)
    validation = run.validate_expected()
    for policy, result in run.policy_evaluations.items():
        print(
            f"[REGION ABLATION 3] {policy}: {result['status']} "
            f"matching={result['maximum_matching_cardinality']} "
            f"classification={result['classification']}"
        )
    print(f"[REGION ABLATION 3] output: {run.run_dir}")
    print(f"[REGION ABLATION 3] expectations passed: {validation['passed']}")
    if not arguments.skip_expectation_validation and not validation["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
