"""Production RGB-D inspection capture for Workshop (W1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None

from mujoco_scenes.geometry_checker import (
    CANONICAL_VIEWPOINT_ROLES,
    camera_intrinsics,
    look_at_camera_rotation,
    validate_camera_view,
)
from mujoco_scenes.workshop_phase1.types import ViewObservation

WORKSHOP_RIG_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "workshop_inspection_rigs.yaml"
)


class ProductionInspectionCapture:
    """Captures calibrated multi-view RGB-D observations without simulator segmentation."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        rig_config_path: Path | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.rig_config_path = rig_config_path or WORKSHOP_RIG_CONFIG_PATH
        self._configuration: dict[str, Any] = {}
        if self.rig_config_path.is_file():
            with open(self.rig_config_path, "r", encoding="utf-8") as f:
                self._configuration = yaml.safe_load(f)
        roles = set(self._configuration.get("camera_slots", {}))
        if roles != set(CANONICAL_VIEWPOINT_ROLES):
            raise ValueError(
                "Workshop rig must configure ISO_LEFT, ISO_RIGHT, and DETAIL"
            )

    def get_stage_rig_config(self, stage_region: str) -> dict[str, Any]:
        """Return rig parameters for the specified inspection stage."""
        regions = self._configuration.get("regions", {})
        if stage_region in regions:
            return regions[stage_region]
        return regions.get("INITIAL", {})

    def capture_stage(
        self,
        scene: Any,
        stage_region: str,
        renderer: Any | None = None,
        capture_segmentation: bool = False,
    ) -> list[ViewObservation]:
        """Capture the three canonical calibrated RGB-D viewpoints."""
        if mujoco is None:
            raise RuntimeError("MuJoCo is not available for capture.")

        regions_cfg = self._configuration.get("regions", {})
        if stage_region not in regions_cfg:
            stage_region = "INITIAL"
        region = regions_cfg[stage_region]

        # Settle simulation if configured
        settle_steps = int(region.get("settle_steps", 0))
        for _ in range(max(0, settle_steps)):
            mujoco.mj_step(scene.model, scene.data)

        camera_slots = self._configuration["camera_slots"]

        target_base = np.asarray(region["target_world_m"], dtype=np.float64)
        rig_position = np.asarray(region["rig_position_world_m"], dtype=np.float64)
        up_world = np.asarray(region["up_world"], dtype=np.float64)
        near_depth = float(region.get("near_depth_m", 0.15))
        far_depth = float(region.get("far_depth_m", 3.0))

        # Backup camera parameters
        original_camera_state: dict[str, dict[str, Any]] = {}
        for logical_name, model_cam_name in camera_slots.items():
            cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, model_cam_name)
            if cam_id < 0:
                continue

            cam_spec = region["cameras"][logical_name]
            position = rig_position + np.asarray(cam_spec["position_offset_m"], dtype=np.float64)
            target = target_base + np.asarray(cam_spec.get("look_at_offset_m", (0.0, 0.0, 0.0)), dtype=np.float64)
            rotation = look_at_camera_rotation(position, target, up_world)

            original_camera_state[logical_name] = {
                "cam_id": cam_id,
                "pos": scene.model.cam_pos[cam_id].copy(),
                "quat": scene.model.cam_quat[cam_id].copy(),
                "mat0": scene.model.cam_mat0[cam_id].copy(),
                "mode": int(scene.model.cam_mode[cam_id]),
                "targetbodyid": int(scene.model.cam_targetbodyid[cam_id]),
                "fovy": float(scene.model.cam_fovy[cam_id]),
            }

            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
            scene.model.cam_pos[cam_id] = position
            scene.model.cam_quat[cam_id] = quaternion
            scene.model.cam_mat0[cam_id] = rotation.reshape(-1)
            scene.model.cam_mode[cam_id] = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
            scene.model.cam_targetbodyid[cam_id] = -1
            scene.model.cam_fovy[cam_id] = float(cam_spec.get("fovy_degrees", 60.0))

        mujoco.mj_forward(scene.model, scene.data)

        should_close_renderer = False
        if renderer is None:
            renderer = mujoco.Renderer(scene.model, height=self.height, width=self.width)
            should_close_renderer = True

        observations: list[ViewObservation] = []

        try:
            for logical_name, model_cam_name in camera_slots.items():
                cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, model_cam_name)
                if cam_id < 0:
                    continue

                # Production: strictly disable segmentation rendering
                renderer.disable_segmentation_rendering()
                renderer.disable_depth_rendering()
                renderer.update_scene(scene.data, camera=cam_id)
                rgb = renderer.render().copy()

                renderer.enable_depth_rendering()
                renderer.update_scene(scene.data, camera=cam_id)
                depth = renderer.render().copy().astype(np.float64)
                renderer.disable_depth_rendering()

                seg = None
                if capture_segmentation:
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(scene.data, camera=cam_id)
                    seg = renderer.render().copy()
                    renderer.disable_segmentation_rendering()

                # Camera transforms
                cam_pos = scene.data.cam_xpos[cam_id].copy()
                cam_mat = scene.data.cam_xmat[cam_id].reshape(3, 3).copy()
                fovy = float(scene.model.cam_fovy[cam_id])
                intrinsics = camera_intrinsics(fovy, self.width, self.height)

                # Validate viewpoint geometry
                validation = validate_camera_view(
                    camera_position=cam_pos,
                    camera_rotation=cam_mat,
                    target_world=target_base,
                    intrinsics=intrinsics,
                    width=self.width,
                    height=self.height,
                    depth_m=depth,
                    near_depth_m=near_depth,
                    far_depth_m=far_depth,
                    maximum_target_angle_degrees=85.0,
                    minimum_valid_depth_pixels=100,
                )

                obs = ViewObservation(
                    camera_id=logical_name,
                    rgb=rgb,
                    depth_m=depth,
                    intrinsics=intrinsics,
                    camera_position_world=cam_pos,
                    camera_rotation_world=cam_mat,
                    validation=validation,
                    segmentation=seg,
                )
                observations.append(obs)
        finally:
            if should_close_renderer:
                renderer.close()

            # Restore camera state
            for state in original_camera_state.values():
                c_id = state["cam_id"]
                scene.model.cam_pos[c_id] = state["pos"]
                scene.model.cam_quat[c_id] = state["quat"]
                scene.model.cam_mat0[c_id] = state["mat0"]
                scene.model.cam_mode[c_id] = state["mode"]
                scene.model.cam_targetbodyid[c_id] = state["targetbodyid"]
                scene.model.cam_fovy[c_id] = state["fovy"]
            mujoco.mj_forward(scene.model, scene.data)

        return observations


class MultiViewCameraRig(ProductionInspectionCapture):
    """Calibrated multi-view camera rig for incremental stage observations."""

    def __init__(
        self,
        scene: Any | None = None,
        width: int = 1280,
        height: int = 720,
        rig_config_path: Path | None = None,
    ) -> None:
        super().__init__(width=width, height=height, rig_config_path=rig_config_path)
        self.scene = scene

    def capture_stage_observations(
        self,
        stage_region: str = "INITIAL",
        scene: Any | None = None,
        capture_segmentation: bool = False,
    ) -> list[ViewObservation]:
        target_scene = scene or self.scene
        if target_scene is None:
            raise ValueError("Scene must be provided to capture_stage_observations")
        return self.capture_stage(
            scene=target_scene,
            stage_region=stage_region,
            capture_segmentation=capture_segmentation,
        )
