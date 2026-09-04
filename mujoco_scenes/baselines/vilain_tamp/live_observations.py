"""Concrete, baseline-owned RGB-D adapters for the benchmark MuJoCo scenes."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .config import Domain, ObservationMode
from .observations import (
    CameraFrameCapture,
    FIXED_INSPECTION_ORDERS,
    ObservationProtocol,
)


CANONICAL_CAMERAS: Mapping[Domain, tuple[str, ...]] = {
    Domain.KITCHEN: (
        "left_shoulder_camera",
        "right_shoulder_camera",
        "overhead_camera",
        "side_camera",
        "front_camera",
    ),
    Domain.LIVING_ROOM: (
        "l2_camera_left",
        "l2_camera_right",
        "l2_camera_top",
        "l2_camera_front",
        "l2_camera_close",
    ),
    Domain.WORKSHOP: (
        "workshop_camera_left",
        "workshop_camera_right",
        "workshop_camera_top",
        "workshop_camera_front",
        "workshop_camera_close",
    ),
}

VIEW_DESCRIPTIONS: Mapping[Domain, Mapping[str, str]] = {
    domain: {
        camera: f"Canonical {domain.value.replace('_', ' ')} view {index + 1}"
        for index, camera in enumerate(cameras)
    }
    for domain, cameras in CANONICAL_CAMERAS.items()
}


class LiveObservationError(RuntimeError):
    """Raised when a live scene cannot satisfy the observation contract."""


@dataclass(frozen=True)
class LiveObservationRuntime:
    """Keep the live scene and its runner-ready protocol together."""

    scene: Any
    protocol: ObservationProtocol


class MuJoCoRGBDCaptureBackend:
    """Capture aligned RGB-D and public calibration from one live scene."""

    def __init__(
        self,
        *,
        domain: Domain | str,
        scene: Any,
        width: int = 640,
        height: int = 480,
        mujoco_module: Any | None = None,
    ) -> None:
        self.domain = domain if isinstance(domain, Domain) else Domain(domain)
        if width <= 0 or height <= 0:
            raise ValueError("capture width and height must be positive")
        if not hasattr(scene, "model") or not hasattr(scene, "data"):
            raise LiveObservationError("scene must expose MuJoCo model and data")
        if mujoco_module is None:
            import mujoco as mujoco_module

        self.scene = scene
        self.width = width
        self.height = height
        self._mujoco = mujoco_module

    def capture(self, camera_id: str, stage_id: str) -> CameraFrameCapture:
        del stage_id
        if camera_id not in CANONICAL_CAMERAS[self.domain]:
            raise LiveObservationError(
                f"camera {camera_id!r} is not canonical for {self.domain.value}"
            )
        camera_index = self._mujoco.mj_name2id(
            self.scene.model,
            self._mujoco.mjtObj.mjOBJ_CAMERA,
            camera_id,
        )
        if camera_index < 0:
            raise LiveObservationError(f"scene camera is missing: {camera_id}")

        self._mujoco.mj_forward(self.scene.model, self.scene.data)
        renderer = self._mujoco.Renderer(
            self.scene.model, height=self.height, width=self.width
        )
        try:
            scene_option = self._scene_option()
            update_arguments = {"camera": camera_index}
            if scene_option is not None:
                update_arguments["scene_option"] = scene_option
            renderer.disable_depth_rendering()
            if hasattr(renderer, "disable_segmentation_rendering"):
                renderer.disable_segmentation_rendering()
            renderer.update_scene(self.scene.data, **update_arguments)
            rgb = np.asarray(renderer.render()).copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(self.scene.data, **update_arguments)
            depth = np.asarray(renderer.render(), dtype=np.float32).copy()
            renderer.disable_depth_rendering()
        finally:
            renderer.close()

        _validate_rendered_arrays(rgb, depth, self.width, self.height)
        intrinsics = _camera_intrinsics(
            float(self.scene.model.cam_fovy[camera_index]),
            self.width,
            self.height,
        )
        extrinsics = _camera_extrinsics(
            np.asarray(self.scene.data.cam_xpos[camera_index], dtype=np.float64),
            np.asarray(self.scene.data.cam_xmat[camera_index], dtype=np.float64),
        )
        return CameraFrameCapture(
            camera_id=camera_id,
            view_description=VIEW_DESCRIPTIONS[self.domain][camera_id],
            rgb_png=_encode_png(rgb),
            depth_m=depth,
            intrinsics=_matrix_tuple(intrinsics),
            extrinsics=_matrix_tuple(extrinsics),
        )

    def _scene_option(self) -> Any | None:
        groups = getattr(self.scene, "perception_render_geom_groups", None)
        if groups is None:
            groups = getattr(self.scene, "perception_geom_groups", None)
        if groups is None or not hasattr(self._mujoco, "MjvOption"):
            return None
        option = self._mujoco.MjvOption()
        option.geomgroup[:] = 0
        for group in groups:
            index = int(group)
            if 0 <= index < len(option.geomgroup):
                option.geomgroup[index] = 1
        return option


class SceneRegionOpeningBackend:
    """Expose only fixed-order, generic scene articulation to inspection."""

    def __init__(self, *, domain: Domain | str, scene: Any) -> None:
        self.domain = domain if isinstance(domain, Domain) else Domain(domain)
        opener = getattr(scene, "open_container", None)
        if FIXED_INSPECTION_ORDERS[self.domain] and not callable(opener):
            raise LiveObservationError("scene has no generic open_container method")
        self.scene = scene

    def open_region(self, region_id: str) -> Mapping[str, Any]:
        if region_id not in FIXED_INSPECTION_ORDERS[self.domain]:
            raise LiveObservationError(
                f"region {region_id!r} is outside the fixed inspection order"
            )
        self.scene.open_container(region_id)
        return {"region_id": region_id, "opened": True}


def create_live_observation_runtime(
    *,
    domain: Domain | str,
    variant: str,
    observation_mode: ObservationMode | str,
    output_root: str | Path,
    robot: str = "none",
    width: int = 640,
    height: int = 480,
    layout_seed: int = 0,
    scene: Any | None = None,
    mujoco_module: Any | None = None,
) -> LiveObservationRuntime:
    """Build one actual benchmark scene and its baseline observation protocol.

    ``variant`` is used only to construct the physical scene. It is not passed
    to :class:`ObservationProtocol` or included in model-facing observations.
    The optional scene and module parameters support side-effect-free contract
    tests; production callers omit them.
    """
    resolved_domain = domain if isinstance(domain, Domain) else Domain(domain)
    resolved_mode = (
        observation_mode
        if isinstance(observation_mode, ObservationMode)
        else ObservationMode(observation_mode)
    )
    if not variant.strip():
        raise ValueError("variant must not be empty")
    live_scene = (
        scene
        if scene is not None
        else _create_scene(
            resolved_domain, variant, robot=robot, layout_seed=layout_seed
        )
    )
    capture = MuJoCoRGBDCaptureBackend(
        domain=resolved_domain,
        scene=live_scene,
        width=width,
        height=height,
        mujoco_module=mujoco_module,
    )
    opening = (
        SceneRegionOpeningBackend(domain=resolved_domain, scene=live_scene)
        if FIXED_INSPECTION_ORDERS[resolved_domain]
        else None
    )
    protocol = ObservationProtocol(
        domain=resolved_domain,
        observation_mode=resolved_mode,
        camera_ids=CANONICAL_CAMERAS[resolved_domain],
        output_root=output_root,
        capture_backend=capture,
        opening_backend=opening,
    )
    return LiveObservationRuntime(live_scene, protocol)


def _create_scene(
    domain: Domain, variant: str, *, robot: str, layout_seed: int
) -> Any:
    from mujoco_scenes.final_paper_variant_labels import (
        VARIANT_LABELS,
        resolve_variant_name,
    )

    internal_variant = resolve_variant_name(domain.value, variant)
    if internal_variant not in VARIANT_LABELS[domain.value]:
        raise ValueError(f"unknown {domain.value} benchmark variant: {variant}")
    if domain is Domain.KITCHEN:
        from mujoco_scenes.scene_loader import KitchenScene

        variant_code = internal_variant.split("_", 1)[0]
        scene_name = f"S1_integrated_kitchen_object_function_feasibility_{variant_code}"
        return KitchenScene(
            scene_name,
            include_robot=robot != "none",
            robot=robot,
            layout_seed=layout_seed,
        )
    if domain is Domain.LIVING_ROOM:
        from mujoco_scenes.living_room_region_scene import L2LivingRoomRegionScene
        from mujoco_scenes.living_room_variants import scene_name

        return L2LivingRoomRegionScene(scene_name(internal_variant), robot=robot)

    from mujoco_scenes.workshop_scene import WorkshopScene

    return WorkshopScene(robot=robot, variant=internal_variant)


def _camera_intrinsics(fovy_degrees: float, width: int, height: int) -> np.ndarray:
    if not 0.0 < fovy_degrees < 180.0:
        raise LiveObservationError(f"invalid camera vertical FOV: {fovy_degrees}")
    focal = 0.5 * height / math.tan(0.5 * math.radians(fovy_degrees))
    return np.asarray(
        ((focal, 0.0, 0.5 * width), (0.0, focal, 0.5 * height), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _camera_extrinsics(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if position.shape != (3,) or rotation.size != 9:
        raise LiveObservationError("camera pose has invalid dimensions")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
        raise LiveObservationError("camera pose contains non-finite values")
    world_from_mujoco_camera = rotation.reshape(3, 3)
    # Observation coordinates use +X right, +Y image-down, +Z forward;
    # MuJoCo camera coordinates use +X right, +Y up, and look along -Z.
    axis_conversion = np.diag((1.0, -1.0, -1.0))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = world_from_mujoco_camera @ axis_conversion
    transform[:3, 3] = position
    return transform


def _encode_png(rgb: np.ndarray) -> bytes:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _validate_rendered_arrays(
    rgb: np.ndarray, depth: np.ndarray, width: int, height: int
) -> None:
    if rgb.shape != (height, width, 3):
        raise LiveObservationError(
            f"RGB render has shape {rgb.shape}; expected {(height, width, 3)}"
        )
    if depth.shape != (height, width):
        raise LiveObservationError(
            f"depth render has shape {depth.shape}; expected {(height, width)}"
        )
    if np.any(np.isfinite(depth) & (depth < 0.0)):
        raise LiveObservationError("depth render contains negative metric values")


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)
