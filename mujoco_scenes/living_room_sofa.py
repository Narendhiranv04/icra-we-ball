"""Low-camera inspection of the bounded region beneath the living-room sofa."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from PIL import Image

from mujoco_scenes.living_room_cameras import SOFA_CAMERAS
from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    camera_intrinsics,
    gate_points_to_volume,
)
from mujoco_scenes.perception import ImageSegmenter, validate_segmentations
from mujoco_scenes.sam3_client import create_segmenter


SOFA_VOLUME_MINIMUM = np.array((-1.20, -1.70, 0.0))
SOFA_VOLUME_MAXIMUM = np.array((-0.45, -1.22, 0.23))
MINIMUM_REGION_PIXELS = 20
MINIMUM_SUPPORTING_CAMERAS = 2


@dataclass(frozen=True)
class SofaCameraEvidence:
    camera_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    position_world_m: np.ndarray
    rotation_world_from_camera: np.ndarray
    intrinsics: np.ndarray
    region_points: np.ndarray


def region_points_from_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
) -> np.ndarray:
    points, _pixels = backproject_masked_depth(
        depth_m,
        mask,
        intrinsics,
        camera_position,
        camera_rotation,
        min_depth=0.03,
        max_depth=2.0,
    )
    inside = gate_points_to_volume(
        points,
        minimum_world_m=SOFA_VOLUME_MINIMUM,
        maximum_world_m=SOFA_VOLUME_MAXIMUM,
        boundary_margin_m=0.0,
    )
    return points[inside]


class SofaInspectionExecutor:
    """Capture five base-mounted views and update only observed scene state."""

    def __init__(
        self,
        scene,
        *,
        perception_mode: str = "oracle",
        segmenter: ImageSegmenter | None = None,
        output_dir: str | Path = "runs/living_room_sofa",
        width: int = 640,
        height: int = 480,
    ) -> None:
        if perception_mode not in {"oracle", "sam3"}:
            raise ValueError("Sofa perception must be oracle or sam3")
        if scene.robot_name != "google":
            raise ValueError("Under-sofa inspection requires Google Robot")
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.perception_mode = perception_mode
        self.segmenter = (
            create_segmenter()
            if perception_mode == "sam3" and segmenter is None
            else segmenter
        )
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.status = "Under-sofa region has not been inspected"
        self.failure: str | None = None
        self.mode = "idle"
        self.last_evidence: tuple[SofaCameraEvidence, ...] = ()

    @property
    def busy(self) -> bool:
        return False

    @property
    def navigation_safe(self) -> bool:
        return True

    def update(self) -> None:
        return

    def progress(self) -> float:
        return 1.0 if self.mode == "complete" else 0.0

    def _oracle_mask(self, segmentation: np.ndarray) -> np.ndarray:
        body_id = self.scene.body_id("remote_control")
        geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
        return (
            segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        ) & np.isin(segmentation[:, :, 0], geom_ids)

    def _learned_mask(self, rgb: np.ndarray, camera_id: str) -> np.ndarray:
        if self.segmenter is None:
            raise RuntimeError("SAM 3 segmenter is not configured")
        instances = validate_segmentations(
            self.segmenter.segment(
                rgb,
                camera_id=camera_id,
                prompts=("remote control",),
            ),
            image_shape=(self.height, self.width),
        )
        if not instances:
            return np.zeros((self.height, self.width), dtype=bool)
        return np.logical_or.reduce([item.mask for item in instances])

    def request_inspect(self, current_location: str) -> None:
        if self.scene.scenario != "lost_remote":
            raise RuntimeError("Load --scenario lost_remote first")
        if current_location != "couch":
            raise RuntimeError("Move to Couch before inspecting beneath it")
        self.failure = None
        self.mode = "capturing"
        renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        evidence: list[SofaCameraEvidence] = []
        try:
            mujoco.mj_forward(self.model, self.data)
            for camera_id in SOFA_CAMERAS:
                model_camera_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
                )
                renderer.update_scene(self.data, camera=model_camera_id)
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                if self.perception_mode == "oracle":
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render().copy()
                    renderer.disable_segmentation_rendering()
                    mask = self._oracle_mask(segmentation)
                else:
                    mask = self._learned_mask(rgb, camera_id)
                intrinsics = camera_intrinsics(
                    float(self.model.cam_fovy[model_camera_id]),
                    self.width,
                    self.height,
                )
                position = self.data.cam_xpos[model_camera_id].copy()
                rotation = self.data.cam_xmat[model_camera_id].reshape(3, 3).copy()
                region_points = region_points_from_mask(
                    depth,
                    mask,
                    intrinsics,
                    position,
                    rotation,
                )
                evidence.append(
                    SofaCameraEvidence(
                        camera_id,
                        rgb,
                        depth,
                        mask,
                        position,
                        rotation,
                        intrinsics,
                        region_points,
                    )
                )
        except Exception as error:
            self.failure = str(error)
            self.mode = "failed"
            self.status = f"Under-sofa inspection failed: {error}"
            raise
        finally:
            renderer.close()

        self.last_evidence = tuple(evidence)
        supporting_cameras = sum(
            len(item.region_points) >= MINIMUM_REGION_PIXELS
            for item in evidence
        )
        detected = supporting_cameras >= MINIMUM_SUPPORTING_CAMERAS
        self.scene.under_sofa_inspected = True
        self.scene.lost_remote_detected = detected
        self.mode = "complete"
        self.status = (
            "Remote observed beneath sofa by "
            f"{supporting_cameras}/{len(SOFA_CAMERAS)} foot cameras"
            if detected
            else (
                "Under-sofa region inspected; remote had support from "
                f"{supporting_cameras}/{len(SOFA_CAMERAS)} cameras "
                f"(requires {MINIMUM_SUPPORTING_CAMERAS})"
            )
        )
        self._export(evidence, supporting_cameras)

    def _export(
        self,
        evidence: Sequence[SofaCameraEvidence],
        supporting_cameras: int,
    ) -> None:
        root = self.output_dir / self.perception_mode
        root.mkdir(parents=True, exist_ok=True)
        cameras = {}
        for item in evidence:
            camera_dir = root / item.camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(item.rgb.astype(np.uint8)).save(camera_dir / "rgb.png")
            overlay = item.rgb.astype(np.float32).copy()
            overlay[item.mask] = (
                0.55 * overlay[item.mask]
                + 0.45 * np.array((50.0, 220.0, 90.0))
            )
            Image.fromarray(overlay.astype(np.uint8)).save(
                camera_dir / "mask_overlay.png"
            )
            cameras[item.camera_id] = {
                "mask_pixels": int(np.count_nonzero(item.mask)),
                "region_points": len(item.region_points),
                "position_world_m": item.position_world_m.tolist(),
                "rotation_world_from_camera": (
                    item.rotation_world_from_camera.tolist()
                ),
                "intrinsics": item.intrinsics.tolist(),
            }
        (root / "inspection.json").write_text(
            json.dumps(
                {
                    "perception_mode": self.perception_mode,
                    "region": "under_sofa",
                    "region_bounds_world_m": {
                        "minimum": SOFA_VOLUME_MINIMUM.tolist(),
                        "maximum": SOFA_VOLUME_MAXIMUM.tolist(),
                    },
                    "supporting_cameras": supporting_cameras,
                    "minimum_supporting_cameras": MINIMUM_SUPPORTING_CAMERAS,
                    "remote_observed": (
                        supporting_cameras >= MINIMUM_SUPPORTING_CAMERAS
                    ),
                    "cameras": cameras,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


__all__ = [
    "SOFA_CAMERAS",
    "SOFA_VOLUME_MAXIMUM",
    "SOFA_VOLUME_MINIMUM",
    "MINIMUM_SUPPORTING_CAMERAS",
    "SofaInspectionExecutor",
    "region_points_from_mask",
]
