"""Run one five-view geometric inspection with Google Robot retained."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from mujoco_scenes.geometry_checker import GeometryChecker
from mujoco_scenes.observed_geometry import ObservedGeometryState
from mujoco_scenes.scene_loader import CONTAINER_JOINTS, KitchenScene


def _load_segmenter(spec: str):
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("segmenter must use the form package.module:factory")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="S1_coffee_missing_mug")
    parser.add_argument(
        "--region", choices=("INITIAL", *CONTAINER_JOINTS), default="INITIAL"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/geometry"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--prompt", action="append", default=[])
    perception = parser.add_mutually_exclusive_group(required=True)
    perception.add_argument(
        "--oracle",
        action="store_true",
        help="use MuJoCo instance segmentation for ground-truth evaluation",
    )
    perception.add_argument(
        "--segmenter",
        metavar="MODULE:FACTORY",
        help="load an image-only ImageSegmenter implementation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.oracle and not args.prompt:
        parser.error("learned segmentation requires at least one --prompt")
    segmenter = None if args.oracle else _load_segmenter(args.segmenter)
    scene = KitchenScene(args.scene, robot="google")
    if args.region != "INITIAL":
        scene.open_container(args.region, steps=500)
    checker = GeometryChecker(
        scene,
        width=args.width,
        height=args.height,
        segmenter=segmenter,
        semantic_prompts=args.prompt,
    )
    stage_name = "initial" if args.region == "INITIAL" else f"after_{args.region}"
    stage_debug = args.output / "captures" / stage_name
    run = checker.run_region_inspection(
        args.region,
        stage_output_dir=stage_debug,
    )
    state = ObservedGeometryState(
        args.output / "observed_state", scene_name=scene.scene_name
    )
    associations = state.update(run, stage_label=stage_name)
    print(f"Perception: {run.inspection.metadata['perception_mode']}")
    print(f"Valid cameras: {run.inspection.quality['valid_camera_count']}/5")
    print(f"Accepted objects: {len(associations)}")
    print(f"Observed state: {state.registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
