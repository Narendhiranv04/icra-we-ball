"""One-command observed search -> witness -> PDDL -> validated plan run."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import yaml

from mujoco_scenes.scene_loader import KitchenScene
from mujoco_scenes.sequential_inspection import run_sequential_inspection
from mujoco_scenes.symbolic_planning import (
    compile_plan_and_save,
    ground_symbolic_sources,
)


DEFAULT_SCENE = "S1_integrated_kitchen_object_function_primary"
DEFAULT_TASK = "configs/s1_integrated_kitchen_object_function.yaml"
DEFAULT_SEQUENCE = ("D1", "D2", "C2", "B1", "C1")
VISUAL_REVISION = (
    Path(__file__).resolve().parent
    / "configs"
    / "s1_integrated_visual_revision.yaml"
)


def run_pipeline(arguments: argparse.Namespace) -> dict:
    if arguments.scene != DEFAULT_SCENE:
        raise ValueError(
            "This milestone's authoritative entry point accepts only "
            f"{DEFAULT_SCENE}; got {arguments.scene}"
        )
    scene = KitchenScene(
        arguments.scene,
        include_robot=False,
        robot="none",
        layout_seed=arguments.layout_seed,
    )
    session = run_sequential_inspection(
        scene,
        arguments.inspect_sequence,
        runs_root=arguments.runs_root,
        run_id=arguments.run_id,
        width=arguments.width,
        height=arguments.height,
        task_requirements=arguments.task_requirements,
        stop_on_complete=True,
        semantic_backend=arguments.semantic_detector,
        semantic_model=arguments.semantic_model,
        semantic_vocabulary_path=arguments.semantic_vocabulary,
        semantic_confidence_threshold=arguments.semantic_confidence_threshold,
        semantic_min_supporting_views=arguments.semantic_min_supporting_views,
        grounding_mode="joint",
        save_semantic_overlays=arguments.save_semantic_overlays,
    )
    close_detector = getattr(session.semantic_detector, "close", None)
    if callable(close_detector):
        close_detector()
    ground_symbolic_sources(
        session.run_dir,
        checkpoint=arguments.semantic_model,
        confidence_threshold=arguments.semantic_confidence_threshold,
    )
    result = compile_plan_and_save(session.run_dir, arguments.task_requirements)
    visual_revision = yaml.safe_load(
        VISUAL_REVISION.read_text(encoding="utf-8")
    )
    (session.run_dir / "scene_visual_revision.json").write_text(
        json.dumps(visual_revision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    witness_path = session.run_dir / "latest_witness.json"
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    (session.run_dir / "functional_witness.json").write_text(
        json.dumps(witness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n" + (session.run_dir / "combined_action_sequence.txt").read_text())
    print(f"Artifacts: {session.run_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-robot observed kitchen search and symbolic planning"
    )
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--no-robot", action="store_true", default=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--run-id",
        default=f"integrated_symbolic_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument("--task-requirements", default=DEFAULT_TASK)
    parser.add_argument(
        "--inspect-sequence", nargs="+", default=list(DEFAULT_SEQUENCE)
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=None,
        help=(
            "Deterministically randomize which three target vessels are "
            "stored one each in C2, B1 and C1"
        ),
    )
    parser.add_argument("--semantic-detector", default="yolo_world")
    parser.add_argument(
        "--semantic-model", default="semantic_model_cache/yolov8m-worldv2.pt"
    )
    parser.add_argument(
        "--semantic-vocabulary",
        default="mujoco_scenes/configs/semantic_vocabulary.yaml",
    )
    parser.add_argument("--semantic-confidence-threshold", type=float, default=0.03)
    parser.add_argument("--semantic-min-supporting-views", type=int, default=2)
    parser.add_argument("--save-semantic-overlays", action="store_true")
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
