"""Capture the workshop with the shared five-view RGB-D pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_scenes.geometry_checker import (
    GeometryChecker,
    PointCloudRun,
    print_run_summary,
    write_ply,
)
from mujoco_scenes.sam3_client import create_segmenter
from mujoco_scenes.workshop_scene import WorkshopScene


DEFAULT_PROMPTS = (
    "protective plate",
    "screwdriver",
    "powered screwdriver",
    "screw",
    "wrench",
    "pliers",
    "wooden frame",
    "parts tray",
)
STAGES = (
    ("INITIAL", "000_initial"),
    ("LEFT_DRAWER", "001_left_drawer"),
    ("RIGHT_DRAWER", "002_right_drawer"),
    ("TOOL_CABINET", "003_tool_cabinet"),
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _export_fused_clouds(run: PointCloudRun, output_dir: Path) -> dict[str, Any]:
    fused_dir = output_dir / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)
    all_points = []
    all_colors = []
    objects = []
    for index, (instance_id, cloud) in enumerate(sorted(run.clouds.items()), 1):
        filename = f"object_{index:04d}.ply"
        write_ply(fused_dir / filename, cloud.points, cloud.colors)
        if len(cloud.points):
            all_points.append(cloud.points)
            all_colors.append(cloud.colors)
        objects.append(
            {
                "debug_instance_id": instance_id,
                "object_kind": cloud.object_kind,
                "point_count": len(cloud.points),
                "contributing_camera_count": sum(
                    count > 0 for count in cloud.pixels_by_camera.values()
                ),
                "ply": f"fused/{filename}",
            }
        )
    combined_path = fused_dir / "all_objects.ply"
    write_ply(
        combined_path,
        np.concatenate(all_points)
        if all_points
        else np.empty((0, 3), dtype=np.float32),
        np.concatenate(all_colors)
        if all_colors
        else np.empty((0, 3), dtype=np.uint8),
    )
    return {
        "objects": objects,
        "accepted_object_count": len(objects),
        "total_point_count": run.total_points,
        "combined_ply": "fused/all_objects.ply",
        "capture_quality": run.inspection.quality if run.inspection else None,
        "timings_seconds": run.timings_seconds,
    }


def run_workshop_pointcloud(
    output_dir: str | Path,
    *,
    robot: str = "google",
    variant: str = "F0_BASE",
    width: int = 640,
    height: int = 480,
    segmentation: str = "oracle",
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
) -> tuple[WorkshopScene, dict[str, Any]]:
    """Capture all workshop regions and return the final live scene."""
    if segmentation not in {"oracle", "sam3"}:
        raise ValueError("segmentation must be oracle or sam3")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Workshop point-cloud run already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    scene = WorkshopScene(robot=robot, variant=variant)
    checker = GeometryChecker(
        scene,
        width=width,
        height=height,
        segmenter=create_segmenter() if segmentation == "sam3" else None,
        semantic_prompts=prompts if segmentation == "sam3" else (),
    )
    stage_records = []
    for region_id, directory_name in STAGES:
        if region_id != "INITIAL":
            scene.open_container(region_id)
        stage_dir = output / directory_name
        try:
            run = checker.run_region_inspection(
                region_id,
                stage_output_dir=stage_dir,
                rig_config=scene.inspection_rig_config,
            )
        finally:
            if region_id != "INITIAL":
                scene.close_container(region_id)
        stage_record = {
            "region_id": region_id,
            "directory": directory_name,
            **_export_fused_clouds(run, stage_dir),
        }
        _write_json(stage_dir / "stage_summary.json", stage_record)
        stage_records.append(stage_record)
        print_run_summary(run)

    manifest = {
        "schema_version": 1,
        "scene": scene.scene_name,
        "variant": variant,
        "robot": robot,
        "segmentation": segmentation,
        "segmentation_scope": (
            "EXPLICIT_MUJOCO_ORACLE_DEBUG"
            if segmentation == "oracle"
            else "IMAGE_ONLY_SAM3"
        ),
        "camera_count": 5,
        "camera_ids": list(scene.point_cloud_cameras),
        "resolution": [width, height],
        "inspection_order": [stage[0] for stage in STAGES],
        "containers_closed_after_capture": True,
        "prompts": list(prompts) if segmentation == "sam3" else [],
        "stages": stage_records,
        "final_region_states": scene.get_region_observation_states(),
        "final_task_state": scene.get_task_scene_state(),
    }
    _write_json(output / "manifest.json", manifest)
    return scene, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("google", "none"), default="google")
    parser.add_argument("--variant", default="F0_BASE")
    parser.add_argument("--segmentation", choices=("oracle", "sam3"), default="oracle")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New output directory; defaults to a timestamped runs directory",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--camera", default="free")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    output = arguments.output or Path("runs/workshop_pointcloud") / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    scene, manifest = run_workshop_pointcloud(
        output,
        robot=arguments.robot,
        variant=arguments.variant,
        width=arguments.width,
        height=arguments.height,
        segmentation=arguments.segmentation,
        prompts=tuple(arguments.prompts or DEFAULT_PROMPTS),
    )
    print(f"\nWorkshop point-cloud run: {Path(output).resolve()}")
    print(f"Stages captured: {len(manifest['stages'])}")
    if arguments.viewer:
        scene.launch_viewer(arguments.camera)


if __name__ == "__main__":
    main()
