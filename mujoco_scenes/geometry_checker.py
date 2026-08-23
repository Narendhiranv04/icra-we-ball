"""Calibrated multi-view RGB-D object reconstruction for MuJoCo scenes."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

try:
    import mujoco
except ModuleNotFoundError:  # Allow the pure geometry helpers to be unit tested.
    mujoco = None


CANONICAL_VIEWPOINT_ROLES = ("ISO_LEFT", "ISO_RIGHT", "DETAIL")
KITCHEN_VIEWPOINT_ROLES = (
    "inspection_left",
    "inspection_right",
    "inspection_top",
    "inspection_front",
    "inspection_close",
)
DEFAULT_FUSION_CAMERAS = (
    "left_shoulder_camera",
    "right_shoulder_camera",
    "overhead_camera",
    "side_camera",
    "front_camera",
)
INSPECTION_RIG_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "inspection_rigs.yaml"
)
MEASUREMENT_EVIDENCE_PURPOSE = "MEASUREMENT_EVIDENCE"
CUMULATIVE_VISUALIZATION_PURPOSE = (
    "CUMULATIVE_VISUALIZATION_NOT_MEASUREMENT"
)


@dataclass
class ObjectPointCloud:
    """Fused world-frame samples belonging to one MuJoCo object body."""

    instance_name: str
    object_kind: str
    points: np.ndarray
    colors: np.ndarray
    pixels_by_camera: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementEvidence:
    """One stage-local, region-gated cloud accepted for measurement.

    Property extraction deliberately requires this type. Historical and
    scene-combined clouds do not carry ``cloud_purpose=MEASUREMENT_EVIDENCE``
    and therefore cannot be passed to the extractor accidentally.
    """

    instance_name: str
    measurement_points: np.ndarray
    measurement_colors: np.ndarray
    contributing_camera_ids: tuple[str, ...]
    points_by_camera: dict[str, np.ndarray]
    source_stage: int | None
    source_region: str
    measurement_cloud_path: str | None
    measurement_quality: dict[str, Any]
    cloud_purpose: str = MEASUREMENT_EVIDENCE_PURPOSE

    def with_provenance(
        self,
        *,
        source_stage: int,
        measurement_cloud_path: str,
    ) -> "MeasurementEvidence":
        return replace(
            self,
            source_stage=source_stage,
            measurement_cloud_path=measurement_cloud_path,
        )


@dataclass
class InspectionCameraCapture:
    """Debug data and validation for one virtual inspection camera."""

    camera_id: str
    model_camera_name: str
    position_world_m: np.ndarray
    rotation_world_from_camera: np.ndarray
    target_world_m: np.ndarray
    intrinsics: np.ndarray
    fovy_degrees: float
    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    validation: dict[str, Any]
    # In-memory only. Keys are converted to generic persistent object IDs
    # before semantic association artifacts are serialized.
    instance_masks: dict[str, np.ndarray] = field(default_factory=dict)
    object_points: dict[str, np.ndarray] = field(default_factory=dict)
    object_colors: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RegionInspection:
    """Fresh evidence and diagnostics for one region-facing stage."""

    region_id: str
    rig_config: dict[str, Any]
    cameras: dict[str, InspectionCameraCapture]
    evidence_clouds: dict[str, MeasurementEvidence]
    rejected_clouds: dict[str, ObjectPointCloud]
    metadata: dict[str, Any]
    quality: dict[str, Any]


@dataclass
class PointCloudRun:
    """Outputs and wall-clock timings from one multi-view reconstruction."""

    clouds: dict[str, ObjectPointCloud]
    cameras: tuple[str, ...]
    width: int
    height: int
    timings_seconds: dict[str, float]
    output_dir: Path | None = None
    inspection: RegionInspection | None = None

    @property
    def total_points(self) -> int:
        return sum(len(cloud.points) for cloud in self.clouds.values())


def camera_intrinsics(fovy_degrees: float, width: int, height: int) -> np.ndarray:
    """Return pinhole intrinsics for MuJoCo's vertical field of view."""
    fy = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_degrees))
    fx = fy
    return np.array(
        [[fx, 0.0, 0.5 * width], [0.0, fy, 0.5 * height], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scale_intrinsics(
    intrinsics: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Scale pinhole intrinsics to a resized, aligned RGB-D pixel grid."""
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def load_inspection_rig_config(
    path: str | Path = INSPECTION_RIG_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate deterministic region-facing rig definitions."""
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or not isinstance(
        config.get("regions"), dict
    ):
        raise ValueError("Inspection-rig configuration must define regions")
    configured_sequence = config.get("inspection_sequence", ())
    if not isinstance(configured_sequence, (list, tuple)):
        raise ValueError("inspection_sequence must be a list when provided")
    required_regions = {"INITIAL", *configured_sequence}
    missing = required_regions - set(config["regions"])
    if missing:
        raise ValueError(
            "Inspection-rig configuration is missing: "
            + ", ".join(sorted(missing))
        )
    camera_slots = config.get("camera_slots", {})
    if len(camera_slots) < 3 or len(set(camera_slots.values())) != len(
        camera_slots
    ):
        raise ValueError(
            "Inspection rig must map at least three distinct viewpoint roles"
        )
    for region_id in required_regions:
        region = config["regions"][region_id]
        cameras = region.get("cameras", {})
        if set(cameras) != set(camera_slots):
            raise ValueError(
                f"Region {region_id} must configure every canonical viewpoint role"
            )
        minimum = np.asarray(
            region["inspection_volume"]["minimum_world_m"], dtype=float
        )
        maximum = np.asarray(
            region["inspection_volume"]["maximum_world_m"], dtype=float
        )
        if (
            minimum.shape != (3,)
            or maximum.shape != (3,)
            or not np.all(np.isfinite(minimum))
            or not np.all(np.isfinite(maximum))
            or np.any(maximum <= minimum)
        ):
            raise ValueError(f"Invalid inspection volume for {region_id}")
    return config


def look_at_camera_rotation(
    camera_position: np.ndarray,
    target_position: np.ndarray,
    up_world: np.ndarray,
) -> np.ndarray:
    """Return MuJoCo camera axes in world coordinates.

    Columns are local +X (right), local +Y (up), and local +Z (backward);
    MuJoCo's camera optical axis is local -Z.
    """
    position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    up = np.asarray(up_world, dtype=np.float64)
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    up_norm = np.linalg.norm(up)
    if (
        not np.all(np.isfinite(position))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(up))
        or forward_norm <= 1e-9
        or up_norm <= 1e-9
    ):
        raise ValueError("Camera look-at pose must be finite and non-degenerate")
    forward /= forward_norm
    up /= up_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-9:
        raise ValueError("Camera look direction cannot be parallel to up")
    right /= right_norm
    corrected_up = np.cross(right, forward)
    rotation = np.column_stack((right, corrected_up, -forward))
    if (
        not np.all(np.isfinite(rotation))
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("Invalid world-from-camera rotation")
    return rotation


def validate_camera_view(
    *,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    target_world: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    depth_m: np.ndarray,
    near_depth_m: float,
    far_depth_m: float,
    maximum_target_angle_degrees: float,
    minimum_valid_depth_pixels: int,
) -> dict[str, Any]:
    """Validate target geometry, frustum projection, depth, and calibration."""
    reasons: list[str] = []
    position = np.asarray(camera_position, dtype=np.float64)
    rotation = np.asarray(camera_rotation, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    transforms_valid = (
        position.shape == (3,)
        and rotation.shape == (3, 3)
        and np.all(np.isfinite(position))
        and np.all(np.isfinite(rotation))
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    )
    if not transforms_valid:
        reasons.append("INVALID_CAMERA_TRANSFORM")
    intrinsics_valid = (
        intrinsics.shape == (3, 3)
        and np.all(np.isfinite(intrinsics))
        and intrinsics[0, 0] > 0
        and intrinsics[1, 1] > 0
        and 0 <= intrinsics[0, 2] <= width
        and 0 <= intrinsics[1, 2] <= height
    )
    if not intrinsics_valid:
        reasons.append("INVALID_INTRINSICS")

    target_in_front = False
    target_in_frustum = False
    angle_degrees = None
    target_pixel = None
    target_depth = None
    if transforms_valid and intrinsics_valid and np.all(np.isfinite(target)):
        direction = target - position
        distance = float(np.linalg.norm(direction))
        local = rotation.T @ direction
        target_depth = float(-local[2])
        target_in_front = bool(target_depth > 0.0)
        if not target_in_front:
            reasons.append("TARGET_BEHIND_CAMERA")
        else:
            pixel_x = (
                intrinsics[0, 0] * local[0] / target_depth
                + intrinsics[0, 2]
            )
            pixel_y = (
                intrinsics[1, 2]
                - intrinsics[1, 1] * local[1] / target_depth
            )
            target_pixel = [float(pixel_x), float(pixel_y)]
            target_in_frustum = bool(
                0.0 <= pixel_x < width and 0.0 <= pixel_y < height
            )
            if not target_in_frustum:
                reasons.append("TARGET_OUTSIDE_FRUSTUM")
            camera_forward = -rotation[:, 2]
            cosine = float(
                np.clip(
                    np.dot(camera_forward, direction / max(distance, 1e-12)),
                    -1.0,
                    1.0,
                )
            )
            angle_degrees = float(np.rad2deg(np.arccos(cosine)))
            if angle_degrees > maximum_target_angle_degrees:
                reasons.append("TARGET_ANGLE_TOO_LARGE")

    valid_depth = (
        np.isfinite(depth_m)
        & (depth_m > near_depth_m)
        & (depth_m <= far_depth_m)
    )
    valid_depth_pixels = int(np.count_nonzero(valid_depth))
    if valid_depth_pixels < minimum_valid_depth_pixels:
        reasons.append("INSUFFICIENT_VALID_DEPTH")
    if np.asarray(depth_m).shape != (height, width):
        reasons.append("DEPTH_RESOLUTION_MISMATCH")

    return {
        "usable": not reasons,
        "reasons": reasons,
        "target_in_front": bool(target_in_front),
        "target_in_frustum": bool(target_in_frustum),
        "target_angle_degrees": angle_degrees,
        "target_pixel": target_pixel,
        "target_depth_m": target_depth,
        "valid_depth_pixels": valid_depth_pixels,
        "intrinsics_match_resolution": bool(intrinsics_valid),
        "camera_transform_valid": bool(transforms_valid),
    }


def erode_binary_mask(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    """Conservatively erode a mask without adding an image dependency."""
    result = np.asarray(mask, dtype=bool).copy()
    if radius_pixels <= 0:
        return result
    padded = np.pad(
        np.asarray(mask, dtype=bool),
        radius_pixels,
        mode="constant",
        constant_values=False,
    )
    height, width = mask.shape
    for row_offset in range(2 * radius_pixels + 1):
        for column_offset in range(2 * radius_pixels + 1):
            result &= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return result


def reject_depth_discontinuities(
    depth_m: np.ndarray,
    mask: np.ndarray,
    *,
    threshold_m: float,
    radius_pixels: int,
) -> np.ndarray:
    """Remove mask-edge pixels whose neighbouring depth is discontinuous."""
    result = np.asarray(mask, dtype=bool).copy()
    if threshold_m <= 0.0 or radius_pixels <= 0:
        return result
    depth = np.asarray(depth_m, dtype=np.float64)
    height, width = depth.shape
    padded_depth = np.pad(depth, radius_pixels, constant_values=np.nan)
    padded_mask = np.pad(
        result, radius_pixels, mode="constant", constant_values=False
    )
    discontinuous = np.zeros_like(result)
    for dy, dx in (
        (-radius_pixels, 0),
        (radius_pixels, 0),
        (0, -radius_pixels),
        (0, radius_pixels),
    ):
        neighbour_depth = padded_depth[
            radius_pixels + dy : radius_pixels + dy + height,
            radius_pixels + dx : radius_pixels + dx + width,
        ]
        neighbour_mask = padded_mask[
            radius_pixels + dy : radius_pixels + dy + height,
            radius_pixels + dx : radius_pixels + dx + width,
        ]
        comparable = (
            result
            & ~neighbour_mask
            & np.isfinite(depth)
            & np.isfinite(neighbour_depth)
        )
        discontinuous |= comparable & (
            np.abs(depth - neighbour_depth) > threshold_m
        )
    return result & ~discontinuous


def gate_points_to_volume(
    points: np.ndarray,
    *,
    minimum_world_m: np.ndarray,
    maximum_world_m: np.ndarray,
    boundary_margin_m: float,
) -> np.ndarray:
    """Return a mask selecting finite points inside one configured volume."""
    samples = np.asarray(points)
    minimum = np.asarray(minimum_world_m, dtype=np.float64) - boundary_margin_m
    maximum = np.asarray(maximum_world_m, dtype=np.float64) + boundary_margin_m
    return (
        np.all(np.isfinite(samples), axis=1)
        & np.all(samples >= minimum, axis=1)
        & np.all(samples <= maximum, axis=1)
    )


def remove_sparse_voxel_outliers(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    voxel_radius_m: float,
    minimum_neighbours: int,
    minimum_input_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove isolated samples using a deterministic neighbouring-voxel test."""
    if (
        len(points) < minimum_input_points
        or voxel_radius_m <= 0.0
        or minimum_neighbours <= 0
    ):
        return points, colors, 0
    keys = np.floor(points / voxel_radius_m).astype(np.int64)
    occupied: dict[tuple[int, int, int], int] = {}
    for key in map(tuple, keys):
        occupied[key] = occupied.get(key, 0) + 1
    keep = np.zeros(len(points), dtype=bool)
    for index, key_array in enumerate(keys):
        key = tuple(int(value) for value in key_array)
        neighbours = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbours += occupied.get(
                        (key[0] + dx, key[1] + dy, key[2] + dz),
                        0,
                    )
        # Exclude the query point itself.
        keep[index] = neighbours - 1 >= minimum_neighbours
    return points[keep], colors[keep], int(np.count_nonzero(~keep))


def backproject_masked_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    *,
    min_depth: float = 0.0,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project selected pixels and return world points plus valid pixel indices.

    MuJoCo cameras look along local -Z with +X right and +Y up. Image row
    coordinates point downward, hence the sign on local Y.
    """
    valid = (
        mask
        & np.isfinite(depth)
        & (depth > max(0.0, min_depth))
        & (depth <= max_depth)
    )
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    z = depth[rows, cols].astype(np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    local = np.column_stack(
        (
            (cols.astype(np.float64) - cx) * z / fx,
            -(rows.astype(np.float64) - cy) * z / fy,
            -z,
        )
    )
    world = local @ np.asarray(camera_rotation, dtype=np.float64).T
    world += np.asarray(camera_position, dtype=np.float64)
    pixels = np.column_stack((rows, cols)).astype(np.int32)
    return world.astype(np.float32), pixels


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first colored point in each world-aligned voxel."""
    if voxel_size <= 0.0 or len(points) == 0:
        return points, colors
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(voxel_keys, axis=0, return_index=True)
    indices.sort()
    return points[indices], colors[indices]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a colored point cloud as a dependency-free binary-little-endian PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    if len(points):
        vertices["x"], vertices["y"], vertices["z"] = points.T
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as output:
        output.write(header)
        vertices.tofile(output)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the colored binary PLY format emitted by :func:`write_ply`."""
    with Path(path).open("rb") as source:
        first_line = source.readline()
        if first_line != b"ply\n":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = None
        binary_little_endian = False
        while True:
            line = source.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            stripped = line.decode("ascii").strip()
            if stripped == "format binary_little_endian 1.0":
                binary_little_endian = True
            elif stripped.startswith("element vertex "):
                vertex_count = int(stripped.rsplit(" ", 1)[1])
            elif stripped == "end_header":
                break
        if not binary_little_endian or vertex_count is None:
            raise ValueError(f"Unsupported PLY encoding: {path}")
        vertices = np.fromfile(
            source,
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
            count=vertex_count,
        )
    points = np.column_stack(
        (vertices["x"], vertices["y"], vertices["z"])
    ).astype(np.float32)
    colors = np.column_stack(
        (vertices["red"], vertices["green"], vertices["blue"])
    ).astype(np.uint8)
    return points, colors


class GeometryChecker:
    """Capture, mask, back-project, and fuse RGB-D observations."""

    def __init__(
        self,
        scene,
        *,
        cameras: Iterable[str] | None = None,
        width: int = 640,
        height: int = 480,
        max_depth: float = 5.0,
        voxel_size: float = 0.003,
        segmenter: Any | None = None,
        semantic_prompts: Sequence[str] = (),
        render_geom_groups: Iterable[int] | None = None,
        instance_geom_groups: Iterable[int] | None = None,
        allowed_geom_groups: Iterable[int] | None = None,
    ):
        if mujoco is None:
            raise RuntimeError("GeometryChecker requires the mujoco package")
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.cameras = tuple(
            cameras
            if cameras is not None
            else getattr(scene, "point_cloud_cameras", DEFAULT_FUSION_CAMERAS)
        )
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.voxel_size = voxel_size
        self.segmenter = segmenter
        self.semantic_prompts = tuple(semantic_prompts)

        if render_geom_groups is not None:
            self.render_geom_groups: tuple[int, ...] | None = tuple(render_geom_groups)
        elif hasattr(scene, "perception_render_geom_groups") and scene.perception_render_geom_groups is not None:
            self.render_geom_groups = tuple(scene.perception_render_geom_groups)
        elif allowed_geom_groups is not None:
            self.render_geom_groups = tuple(allowed_geom_groups)
        elif hasattr(scene, "perception_geom_groups") and scene.perception_geom_groups is not None:
            self.render_geom_groups = tuple(scene.perception_geom_groups)
        else:
            self.render_geom_groups = None

        if instance_geom_groups is not None:
            self.instance_geom_groups: tuple[int, ...] | None = tuple(instance_geom_groups)
        elif hasattr(scene, "perception_instance_geom_groups") and scene.perception_instance_geom_groups is not None:
            self.instance_geom_groups = tuple(scene.perception_instance_geom_groups)
        elif allowed_geom_groups is not None:
            self.instance_geom_groups = tuple(allowed_geom_groups)
        elif hasattr(scene, "perception_geom_groups") and scene.perception_geom_groups is not None:
            self.instance_geom_groups = tuple(scene.perception_geom_groups)
        else:
            self.instance_geom_groups = None

        self.allowed_geom_groups = self.instance_geom_groups
        self._camera_ids = {
            name: self._require_id(mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in self.cameras
        }

    def _build_scene_option(self) -> mujoco.MjvOption:
        vopt = mujoco.MjvOption()
        if self.render_geom_groups is not None:
            vopt.geomgroup[:] = 0
            for g in self.render_geom_groups:
                if 0 <= g < len(vopt.geomgroup):
                    vopt.geomgroup[g] = 1
        return vopt

    def _require_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _geom_ids_by_instance(
        self,
        instance_names: Iterable[str],
        allowed_geom_groups: Iterable[int] | None = None,
    ) -> dict[str, np.ndarray]:
        root_by_body: dict[int, str] = {}
        for name in instance_names:
            body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, name)
            root_by_body[body_id] = name

        groups = (
            tuple(allowed_geom_groups)
            if allowed_geom_groups is not None
            else self.instance_geom_groups
        )

        geom_ids: dict[str, list[int]] = {name: [] for name in instance_names}
        for geom_id, geom_body_id in enumerate(self.model.geom_bodyid):
            if groups is not None:
                if self.model.geom_group[geom_id] not in groups:
                    continue
            body_id = int(geom_body_id)
            while body_id > 0:
                instance_name = root_by_body.get(body_id)
                if instance_name is not None:
                    geom_ids[instance_name].append(geom_id)
                    break
                body_id = int(self.model.body_parentid[body_id])
        return {
            name: np.asarray(ids, dtype=np.int32) for name, ids in geom_ids.items()
        }

    def run(self, output_dir: str | Path | None = None) -> PointCloudRun:
        """Run one reconstruction over all currently visible object instances."""
        started = time.perf_counter()
        mujoco.mj_forward(self.model, self.data)
        if hasattr(self.scene, "privileged_get_visible_backend_instances"):
            visible = self.scene.privileged_get_visible_backend_instances()
        else:
            visible = self.scene.get_visible_object_instances()
        instance_kinds = {name: kind for name, kind in visible}
        geom_ids = self._geom_ids_by_instance(
            instance_kinds, allowed_geom_groups=self.allowed_geom_groups
        )
        identification_done = time.perf_counter()

        points: dict[str, list[np.ndarray]] = {name: [] for name in instance_kinds}
        colors: dict[str, list[np.ndarray]] = {name: [] for name in instance_kinds}
        pixel_counts: dict[str, dict[str, int]] = {
            name: {} for name in instance_kinds
        }
        render_seconds = 0.0
        projection_seconds = 0.0

        renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        scene_option = self._build_scene_option()
        try:
            for camera_name in self.cameras:
                camera_id = self._camera_ids[camera_name]
                renderer.update_scene(
                    self.data, camera=camera_id, scene_option=scene_option
                )

                render_started = time.perf_counter()
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                render_seconds += time.perf_counter() - render_started

                projection_started = time.perf_counter()
                intrinsics = camera_intrinsics(
                    float(self.model.cam_fovy[camera_id]), self.width, self.height
                )
                camera_position = self.data.cam_xpos[camera_id].copy()
                camera_rotation = self.data.cam_xmat[camera_id].reshape(3, 3).copy()
                is_geom = segmentation[:, :, 1] == int(
                    mujoco.mjtObj.mjOBJ_GEOM
                )
                for instance_name, ids in geom_ids.items():
                    visible_instance_mask = is_geom & np.isin(
                        segmentation[:, :, 0], ids
                    )
                    mask = visible_instance_mask
                    mask = reject_depth_discontinuities(
                        depth,
                        mask,
                        threshold_m=0.060,
                        radius_pixels=1,
                    )
                    world_points, pixels = backproject_masked_depth(
                        depth,
                        mask,
                        intrinsics,
                        camera_position,
                        camera_rotation,
                        max_depth=self.max_depth,
                    )
                    pixel_counts[instance_name][camera_name] = len(world_points)
                    if len(world_points):
                        points[instance_name].append(world_points)
                        colors[instance_name].append(
                            rgb[pixels[:, 0], pixels[:, 1]].astype(np.uint8)
                        )
                projection_seconds += time.perf_counter() - projection_started
        finally:
            renderer.close()

        fusion_started = time.perf_counter()
        clouds: dict[str, ObjectPointCloud] = {}
        for instance_name, object_kind in instance_kinds.items():
            fused_points = (
                np.concatenate(points[instance_name])
                if points[instance_name]
                else np.empty((0, 3), dtype=np.float32)
            )
            fused_colors = (
                np.concatenate(colors[instance_name])
                if colors[instance_name]
                else np.empty((0, 3), dtype=np.uint8)
            )
            fused_points, fused_colors = voxel_downsample(
                fused_points, fused_colors, self.voxel_size
            )
            clouds[instance_name] = ObjectPointCloud(
                instance_name=instance_name,
                object_kind=object_kind,
                points=fused_points,
                colors=fused_colors,
                pixels_by_camera=pixel_counts[instance_name],
            )
        fusion_seconds = time.perf_counter() - fusion_started

        export_started = time.perf_counter()
        resolved_output = Path(output_dir).resolve() if output_dir else None
        run = PointCloudRun(
            clouds=clouds,
            cameras=self.cameras,
            width=self.width,
            height=self.height,
            timings_seconds={
                "visible_object_identification": identification_done - started,
                "rgbd_and_mask_rendering": render_seconds,
                "backprojection_and_world_transform": projection_seconds,
                "fusion_and_voxel_downsampling": fusion_seconds,
            },
            output_dir=resolved_output,
        )
        if resolved_output is not None:
            self._export(run, resolved_output)
        run.timings_seconds["export"] = time.perf_counter() - export_started
        run.timings_seconds["total"] = time.perf_counter() - started
        if resolved_output is not None:
            self._write_manifest(run, resolved_output)
        return run

    def run_region_inspection(
        self,
        region_id: str,
        *,
        stage_output_dir: str | Path | None = None,
        rig_config: str | Path | dict[str, Any] | None = None,
    ) -> PointCloudRun:
        """Capture one fresh, region-facing calibrated measurement stage.

        Unlike :meth:`run`, this path positions a virtual camera rig, validates
        every view, gates points to the requested region, and emits typed
        :class:`MeasurementEvidence` objects. It never reads a historical PLY.
        """
        started = time.perf_counter()
        configuration = (
            load_inspection_rig_config(rig_config)
            if isinstance(rig_config, (str, Path))
            else rig_config
            or load_inspection_rig_config(
                getattr(
                    self.scene,
                    "inspection_rig_config_path",
                    INSPECTION_RIG_CONFIG_PATH,
                )
            )
        )
        if region_id not in configuration["regions"]:
            raise ValueError(f"No inspection rig configured for {region_id}")
        region = configuration["regions"][region_id]
        settle_steps = int(region.get("settle_steps", 0))
        for _ in range(max(0, settle_steps)):
            mujoco.mj_step(self.model, self.data)
        if hasattr(self.scene, "privileged_get_visible_backend_instances"):
            visible = self.scene.privileged_get_visible_backend_instances()
        else:
            visible = self.scene.get_visible_object_instances()
        instance_kinds = {name: kind for name, kind in visible}
        geom_ids = self._geom_ids_by_instance(
            instance_kinds, allowed_geom_groups=self.allowed_geom_groups
        )
        identification_done = time.perf_counter()

        camera_slots = configuration["camera_slots"]
        camera_ids = {
            logical_name: self._require_id(
                mujoco.mjtObj.mjOBJ_CAMERA, model_name
            )
            for logical_name, model_name in camera_slots.items()
        }
        original_camera_state: dict[str, dict[str, Any]] = {}
        target_base = np.asarray(region["target_world_m"], dtype=np.float64)
        rig_position = np.asarray(
            region["rig_position_world_m"], dtype=np.float64
        )
        up_world = np.asarray(region["up_world"], dtype=np.float64)
        configured_poses: dict[str, dict[str, np.ndarray | float]] = {}
        for logical_name, camera_id in camera_ids.items():
            camera = region["cameras"][logical_name]
            position = rig_position + np.asarray(
                camera["position_offset_m"], dtype=np.float64
            )
            target = target_base + np.asarray(
                camera.get("look_at_offset_m", (0.0, 0.0, 0.0)),
                dtype=np.float64,
            )
            rotation = look_at_camera_rotation(position, target, up_world)
            original_camera_state[logical_name] = {
                "pos": self.model.cam_pos[camera_id].copy(),
                "quat": self.model.cam_quat[camera_id].copy(),
                "mat0": self.model.cam_mat0[camera_id].copy(),
                "mode": int(self.model.cam_mode[camera_id]),
                "targetbodyid": int(self.model.cam_targetbodyid[camera_id]),
                "fovy": float(self.model.cam_fovy[camera_id]),
            }
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
            self.model.cam_pos[camera_id] = position
            self.model.cam_quat[camera_id] = quaternion
            self.model.cam_mat0[camera_id] = rotation.reshape(-1)
            self.model.cam_mode[camera_id] = int(
                mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            )
            self.model.cam_targetbodyid[camera_id] = -1
            self.model.cam_fovy[camera_id] = float(
                camera.get("fovy_degrees", 60.0)
            )
            configured_poses[logical_name] = {
                "position": position,
                "target": target,
                "rotation": rotation,
                "fovy": float(self.model.cam_fovy[camera_id]),
            }
        mujoco.mj_forward(self.model, self.data)

        raw_points: dict[str, dict[str, list[np.ndarray]]] = {
            name: {"inside": [], "all": []} for name in instance_kinds
        }
        raw_colors: dict[str, dict[str, list[np.ndarray]]] = {
            name: {"inside": [], "all": []} for name in instance_kinds
        }
        points_by_camera: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in instance_kinds
        }
        pixel_counts: dict[str, dict[str, int]] = {
            name: {} for name in instance_kinds
        }
        captures: dict[str, InspectionCameraCapture] = {}
        render_seconds = 0.0
        projection_seconds = 0.0
        near_depth = float(region["near_depth_m"])
        far_depth = float(region["far_depth_m"])
        processing = configuration.get("processing", {})
        erosion_pixels = int(processing.get("mask_erosion_pixels", 0))
        discontinuity = float(
            processing.get("depth_discontinuity_m", 0.060)
        )
        discontinuity_radius = int(
            processing.get("depth_discontinuity_radius_pixels", 1)
        )
        volume = region["inspection_volume"]
        volume_minimum = np.asarray(
            volume["minimum_world_m"], dtype=np.float64
        )
        volume_maximum = np.asarray(
            volume["maximum_world_m"], dtype=np.float64
        )
        margin = float(region.get("boundary_margin_m", 0.0))
        validation_config = configuration.get("view_validation", {})

        renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        scene_option = self._build_scene_option()
        try:
            for logical_name, model_camera_name in camera_slots.items():
                camera_id = camera_ids[logical_name]
                renderer.update_scene(
                    self.data, camera=camera_id, scene_option=scene_option
                )
                render_started = time.perf_counter()
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                render_seconds += time.perf_counter() - render_started
                intrinsics = camera_intrinsics(
                    float(self.model.cam_fovy[camera_id]),
                    self.width,
                    self.height,
                )

                if (
                    rgb.shape[:2] != (self.height, self.width)
                    or depth.shape != (self.height, self.width)
                    or segmentation.shape[:2] != (self.height, self.width)
                ):
                    validation = {
                        "usable": False,
                        "reasons": ["RGB_DEPTH_SEGMENTATION_GRID_MISMATCH"],
                    }
                else:
                    validation = validate_camera_view(
                        camera_position=self.data.cam_xpos[camera_id].copy(),
                        camera_rotation=self.data.cam_xmat[camera_id]
                        .reshape(3, 3)
                        .copy(),
                        target_world=np.asarray(
                            configured_poses[logical_name]["target"]
                        ),
                        intrinsics=intrinsics,
                        width=self.width,
                        height=self.height,
                        depth_m=depth,
                        near_depth_m=near_depth,
                        far_depth_m=far_depth,
                        maximum_target_angle_degrees=float(
                            validation_config.get(
                                "maximum_target_angle_degrees", 8.0
                            )
                        ),
                        minimum_valid_depth_pixels=int(
                            validation_config.get(
                                "minimum_valid_depth_pixels", 200
                            )
                        ),
                    )
                capture = InspectionCameraCapture(
                    camera_id=logical_name,
                    model_camera_name=model_camera_name,
                    position_world_m=self.data.cam_xpos[camera_id].copy(),
                    rotation_world_from_camera=self.data.cam_xmat[camera_id]
                    .reshape(3, 3)
                    .copy(),
                    target_world_m=np.asarray(
                        configured_poses[logical_name]["target"]
                    ).copy(),
                    intrinsics=intrinsics.copy(),
                    fovy_degrees=float(self.model.cam_fovy[camera_id]),
                    rgb=rgb,
                    depth_m=depth,
                    segmentation=segmentation,
                    validation=validation,
                )
                captures[logical_name] = capture
                if not validation.get("usable", False):
                    continue

                projection_started = time.perf_counter()
                is_geom = segmentation[:, :, 1] == int(
                    mujoco.mjtObj.mjOBJ_GEOM
                )
                for instance_name, ids in geom_ids.items():
                    visible_instance_mask = is_geom & np.isin(
                        segmentation[:, :, 0], ids
                    )
                    # Preserve the visible, unprocessed instance pixels for the
                    # independent RGB detector-to-instance association path.
                    # Geometry filtering below must not alter the association
                    # mask or leak into semantic inference.
                    capture.instance_masks[instance_name] = (
                        visible_instance_mask.copy()
                    )
                    mask = visible_instance_mask
                    mask = erode_binary_mask(mask, erosion_pixels)
                    mask = reject_depth_discontinuities(
                        depth,
                        mask,
                        threshold_m=discontinuity,
                        radius_pixels=discontinuity_radius,
                    )
                    world_points, pixels = backproject_masked_depth(
                        depth,
                        mask,
                        intrinsics,
                        capture.position_world_m,
                        capture.rotation_world_from_camera,
                        min_depth=near_depth,
                        max_depth=far_depth,
                    )
                    inside_mask = gate_points_to_volume(
                        world_points,
                        minimum_world_m=volume_minimum,
                        maximum_world_m=volume_maximum,
                        boundary_margin_m=margin,
                    )
                    inside_points = world_points[inside_mask]
                    inside_pixels = pixels[inside_mask]
                    all_colors = (
                        rgb[pixels[:, 0], pixels[:, 1]].astype(np.uint8)
                        if len(pixels)
                        else np.empty((0, 3), dtype=np.uint8)
                    )
                    inside_colors = (
                        rgb[
                            inside_pixels[:, 0],
                            inside_pixels[:, 1],
                        ].astype(np.uint8)
                        if len(inside_pixels)
                        else np.empty((0, 3), dtype=np.uint8)
                    )
                    raw_points[instance_name]["all"].append(world_points)
                    raw_points[instance_name]["inside"].append(inside_points)
                    raw_colors[instance_name]["all"].append(all_colors)
                    raw_colors[instance_name]["inside"].append(inside_colors)
                    points_by_camera[instance_name][
                        logical_name
                    ] = inside_points
                    pixel_counts[instance_name][logical_name] = len(
                        inside_points
                    )
                    capture.object_points[instance_name] = inside_points
                    capture.object_colors[instance_name] = inside_colors
                projection_seconds += time.perf_counter() - projection_started
        finally:
            renderer.close()
            for logical_name, camera_id in camera_ids.items():
                original = original_camera_state[logical_name]
                self.model.cam_pos[camera_id] = original["pos"]
                self.model.cam_quat[camera_id] = original["quat"]
                self.model.cam_mat0[camera_id] = original["mat0"]
                self.model.cam_mode[camera_id] = original["mode"]
                self.model.cam_targetbodyid[camera_id] = original[
                    "targetbodyid"
                ]
                self.model.cam_fovy[camera_id] = original["fovy"]
            mujoco.mj_forward(self.model, self.data)

        fusion_started = time.perf_counter()
        valid_rig_cameras = tuple(
            camera_id
            for camera_id, capture in captures.items()
            if capture.validation.get("usable", False)
        )
        minimum_rig_cameras = int(
            validation_config.get(
                "minimum_valid_rig_cameras", len(CANONICAL_VIEWPOINT_ROLES)
            )
        )
        minimum_object_cameras = int(
            validation_config.get("minimum_object_camera_count", 2)
        )
        acceptance = configuration.get("evidence_acceptance", {})
        minimum_object_points = int(
            acceptance.get("minimum_object_points", 20)
        )
        minimum_inside_fraction = float(
            acceptance.get("minimum_inside_fraction", 0.55)
        )
        outlier = processing.get("outlier_removal", {})
        outlier_enabled = bool(outlier.get("enabled", False))

        evidence_clouds: dict[str, MeasurementEvidence] = {}
        rejected_clouds: dict[str, ObjectPointCloud] = {}
        accepted_object_names: list[str] = []
        rejected_object_reasons: dict[str, list[str]] = {}
        clouds: dict[str, ObjectPointCloud] = {}
        for instance_name, object_kind in instance_kinds.items():
            all_current_points = (
                np.concatenate(raw_points[instance_name]["all"])
                if raw_points[instance_name]["all"]
                else np.empty((0, 3), dtype=np.float32)
            )
            inside_points = (
                np.concatenate(raw_points[instance_name]["inside"])
                if raw_points[instance_name]["inside"]
                else np.empty((0, 3), dtype=np.float32)
            )
            inside_colors = (
                np.concatenate(raw_colors[instance_name]["inside"])
                if raw_colors[instance_name]["inside"]
                else np.empty((0, 3), dtype=np.uint8)
            )
            all_current_colors = (
                np.concatenate(raw_colors[instance_name]["all"])
                if raw_colors[instance_name]["all"]
                else np.empty((0, 3), dtype=np.uint8)
            )
            inside_fraction = (
                float(len(inside_points) / len(all_current_points))
                if len(all_current_points)
                else 0.0
            )
            inside_points, inside_colors = voxel_downsample(
                inside_points, inside_colors, self.voxel_size
            )
            removed_outliers = 0
            if outlier_enabled:
                (
                    inside_points,
                    inside_colors,
                    removed_outliers,
                ) = remove_sparse_voxel_outliers(
                    inside_points,
                    inside_colors,
                    voxel_radius_m=float(
                        outlier.get("voxel_radius_m", 0.012)
                    ),
                    minimum_neighbours=int(
                        outlier.get("minimum_neighbours", 2)
                    ),
                    minimum_input_points=int(
                        outlier.get("minimum_input_points", 40)
                    ),
                )
            contributing = tuple(
                camera_id
                for camera_id in valid_rig_cameras
                if len(points_by_camera[instance_name].get(camera_id, ())) > 0
            )
            rejection_reasons = []
            if hasattr(self.scene, "inspection_source_region"):
                expected_source_region = self.scene.inspection_source_region(
                    region_id
                )
            else:
                expected_source_region = (
                    "countertop" if region_id == "INITIAL" else region_id
                )
            controller_source_region = None
            if hasattr(self.scene, "get_instance_source_region"):
                controller_source_region = (
                    self.scene.get_instance_source_region(instance_name)
                )
            if (
                expected_source_region is not None
                and controller_source_region is not None
                and controller_source_region != expected_source_region
            ):
                rejection_reasons.append("SOURCE_REGION_MISMATCH")
            if len(inside_points) < minimum_object_points:
                rejection_reasons.append("INSUFFICIENT_POINTS_INSIDE_VOLUME")
            if inside_fraction < minimum_inside_fraction:
                rejection_reasons.append("INSUFFICIENT_INSIDE_FRACTION")
            accepted = not rejection_reasons
            quality_reasons = []
            if len(valid_rig_cameras) < minimum_rig_cameras:
                quality_reasons.append("INSUFFICIENT_VALID_RIG_CAMERAS")
            if len(contributing) < minimum_object_cameras:
                quality_reasons.append("INSUFFICIENT_OBJECT_CAMERA_COVERAGE")
            if len(inside_points) < minimum_object_points:
                quality_reasons.append("INSUFFICIENT_OBJECT_POINTS")
            quality = {
                "quality_is_valid": accepted and not quality_reasons,
                "status": (
                    "VALID" if accepted and not quality_reasons else "INVALID"
                ),
                "reasons": sorted(set(quality_reasons + rejection_reasons)),
                "point_count": len(inside_points),
                "raw_inside_point_count": sum(
                    len(points)
                    for points in raw_points[instance_name]["inside"]
                ),
                "raw_current_point_count": len(all_current_points),
                "inside_fraction": inside_fraction,
                "controller_source_region": controller_source_region,
                "expected_source_region": expected_source_region,
                "valid_rig_camera_count": len(valid_rig_cameras),
                "contributing_camera_count": len(contributing),
                "contributing_camera_ids": list(contributing),
                "outlier_points_removed": removed_outliers,
                "cloud_purpose": MEASUREMENT_EVIDENCE_PURPOSE,
            }
            current_cloud = ObjectPointCloud(
                instance_name=instance_name,
                object_kind=object_kind,
                points=inside_points,
                colors=inside_colors,
                pixels_by_camera=pixel_counts[instance_name],
            )
            if accepted:
                accepted_object_names.append(instance_name)
                evidence_clouds[instance_name] = MeasurementEvidence(
                    instance_name=instance_name,
                    measurement_points=inside_points,
                    measurement_colors=inside_colors,
                    contributing_camera_ids=contributing,
                    points_by_camera={
                        key: value
                        for key, value in points_by_camera[
                            instance_name
                        ].items()
                        if len(value)
                    },
                    source_stage=None,
                    source_region=region_id,
                    measurement_cloud_path=None,
                    measurement_quality=quality,
                )
                clouds[instance_name] = current_cloud
            elif len(all_current_points):
                rejected_object_reasons[instance_name] = rejection_reasons
                rejected_clouds[instance_name] = ObjectPointCloud(
                    instance_name=instance_name,
                    object_kind=object_kind,
                    points=all_current_points,
                    colors=all_current_colors,
                    pixels_by_camera=pixel_counts[instance_name],
                )
        fusion_seconds = time.perf_counter() - fusion_started

        region_state = (
            {"region_id": "INITIAL", "open": True, "inspected": True}
            if region_id == "INITIAL"
            else deepcopy(
                self.scene.get_region_observation_states().get(region_id, {})
            )
        )
        metadata = {
            "region_id": region_id,
            "region_open": bool(region_state.get("open", False)),
            "region_state": region_state,
            "rig_pose": {
                "position_world_m": rig_position.tolist(),
                "target_world_m": target_base.tolist(),
                "up_world": up_world.tolist(),
            },
            "capture_resolution": [self.width, self.height],
            "inspection_volume": {
                "minimum_world_m": volume_minimum.tolist(),
                "maximum_world_m": volume_maximum.tolist(),
                "boundary_margin_m": margin,
                "purpose": "CURRENT_INSPECTION_EVIDENCE_SELECTION_ONLY",
            },
            "near_depth_m": near_depth,
            "far_depth_m": far_depth,
            "settle_steps": settle_steps,
            "valid_cameras": list(valid_rig_cameras),
            "rejected_cameras": {
                camera_id: capture.validation.get("reasons", [])
                for camera_id, capture in captures.items()
                if not capture.validation.get("usable", False)
            },
            "camera_poses": {
                camera_id: {
                    "model_camera_name": capture.model_camera_name,
                    "position_world_m": capture.position_world_m.tolist(),
                    "rotation_world_from_camera": (
                        capture.rotation_world_from_camera.tolist()
                    ),
                    "target_world_m": capture.target_world_m.tolist(),
                    "intrinsics": capture.intrinsics.tolist(),
                    "fovy_degrees": capture.fovy_degrees,
                    "validation": capture.validation,
                }
                for camera_id, capture in captures.items()
            },
        }
        quality = {
            "region_id": region_id,
            "minimum_valid_rig_cameras": minimum_rig_cameras,
            "valid_camera_count": len(valid_rig_cameras),
            "capture_quality_is_valid": (
                len(valid_rig_cameras) >= minimum_rig_cameras
            ),
            "accepted_instance_count": len(accepted_object_names),
            "rejected_instance_count": len(rejected_object_reasons),
            # Instance names are intentionally kept in memory only. The
            # observed-state layer replaces them with generic IDs/tokens.
            "_accepted_instance_names": accepted_object_names,
            "_rejected_instance_reasons": rejected_object_reasons,
        }
        inspection = RegionInspection(
            region_id=region_id,
            rig_config=region,
            cameras=captures,
            evidence_clouds=evidence_clouds,
            rejected_clouds=rejected_clouds,
            metadata=metadata,
            quality=quality,
        )
        resolved_output = (
            Path(stage_output_dir).resolve() if stage_output_dir else None
        )
        run = PointCloudRun(
            clouds=clouds,
            cameras=tuple(camera_slots),
            width=self.width,
            height=self.height,
            timings_seconds={
                "settle_and_visible_identification": (
                    identification_done - started
                ),
                "rgbd_segmentation_and_view_validation": render_seconds,
                "backprojection_world_transform_and_region_gating": (
                    projection_seconds
                ),
                "stage_local_fusion_and_filtering": fusion_seconds,
            },
            output_dir=resolved_output,
            inspection=inspection,
        )
        export_started = time.perf_counter()
        if resolved_output is not None:
            self._export_inspection_debug(run, resolved_output)
        run.timings_seconds["debug_export"] = (
            time.perf_counter() - export_started
        )
        run.timings_seconds["total"] = time.perf_counter() - started
        return run

    def _export_inspection_debug(
        self, run: PointCloudRun, stage_output_dir: Path
    ) -> None:
        """Save per-camera images/clouds without serializing instance names."""
        inspection = run.inspection
        if inspection is None:
            raise ValueError("Inspection debug export requires an inspection")
        camera_root = stage_output_dir / "cameras"
        for camera_id, capture in inspection.cameras.items():
            directory = camera_root / camera_id
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(capture.rgb.astype(np.uint8)).save(
                directory / "rgb.png"
            )
            depth_mm = np.zeros(capture.depth_m.shape, dtype=np.uint16)
            valid_depth = (
                np.isfinite(capture.depth_m) & (capture.depth_m > 0.0)
            )
            depth_mm[valid_depth] = np.clip(
                capture.depth_m[valid_depth] * 1000.0,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            Image.fromarray(depth_mm).save(
                directory / "depth.png"
            )
            segmentation_id = capture.segmentation[:, :, 0].astype(np.int64)
            segmentation_type = capture.segmentation[:, :, 1].astype(np.int64)
            segmentation_rgb = np.zeros(
                (*segmentation_id.shape, 3), dtype=np.uint8
            )
            valid = segmentation_id >= 0
            segmentation_rgb[valid, 0] = (
                segmentation_id[valid] * 53 + segmentation_type[valid] * 17
            ) % 255
            segmentation_rgb[valid, 1] = (
                segmentation_id[valid] * 97 + segmentation_type[valid] * 31
            ) % 255
            segmentation_rgb[valid, 2] = (
                segmentation_id[valid] * 193 + segmentation_type[valid] * 11
            ) % 255
            Image.fromarray(segmentation_rgb).save(
                directory / "segmentation.png"
            )
            camera_points = [
                points
                for points in capture.object_points.values()
                if len(points)
            ]
            camera_colors = [
                capture.object_colors[name]
                for name, points in capture.object_points.items()
                if len(points)
            ]
            write_ply(
                directory / "cloud.ply",
                (
                    np.concatenate(camera_points)
                    if camera_points
                    else np.empty((0, 3), dtype=np.float32)
                ),
                (
                    np.concatenate(camera_colors)
                    if camera_colors
                    else np.empty((0, 3), dtype=np.uint8)
                ),
            )
            (directory / "camera_metadata.json").write_text(
                json.dumps(
                    {
                        "camera_id": camera_id,
                        "model_camera_name": capture.model_camera_name,
                        "position_world_m": (
                            capture.position_world_m.tolist()
                        ),
                        "rotation_world_from_camera": (
                            capture.rotation_world_from_camera.tolist()
                        ),
                        "target_world_m": (
                            capture.target_world_m.tolist()
                        ),
                        "intrinsics": capture.intrinsics.tolist(),
                        "resolution": [run.width, run.height],
                        "depth_png_unit": "mm",
                        "validation": capture.validation,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def _export(self, run: PointCloudRun, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_points, all_colors = [], []
        for cloud in run.clouds.values():
            write_ply(
                output_dir / f"{cloud.instance_name}.ply",
                cloud.points,
                cloud.colors,
            )
            if len(cloud.points):
                all_points.append(cloud.points)
                all_colors.append(cloud.colors)
        write_ply(
            output_dir / "all_visible_objects.ply",
            np.concatenate(all_points) if all_points else np.empty((0, 3), np.float32),
            np.concatenate(all_colors) if all_colors else np.empty((0, 3), np.uint8),
        )

    @staticmethod
    def _write_manifest(run: PointCloudRun, output_dir: Path) -> None:
        payload = {
            "cameras": list(run.cameras),
            "resolution": [run.width, run.height],
            "total_points": run.total_points,
            "timings_seconds": run.timings_seconds,
            "objects": {
                name: {
                    "object_kind": cloud.object_kind,
                    "points": len(cloud.points),
                    "pixels_by_camera": cloud.pixels_by_camera,
                    "ply": f"{name}.ply",
                }
                for name, cloud in run.clouds.items()
            },
            "combined_ply": "all_visible_objects.ply",
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def print_run_summary(run: PointCloudRun) -> None:
    """Print a concise benchmark report."""
    print("\n[GEOMETRY] Five-view point-cloud reconstruction complete")
    print(f"  Cameras: {', '.join(run.cameras)}")
    print(f"  Resolution: {run.width}x{run.height}")
    for name, cloud in run.clouds.items():
        observed = sum(count > 0 for count in cloud.pixels_by_camera.values())
        print(f"  {name}: {len(cloud.points):,} points from {observed}/5 cameras")
    print(f"  Total fused points: {run.total_points:,}")
    for stage, seconds in run.timings_seconds.items():
        print(f"  {stage}: {seconds:.4f} s")
    if run.output_dir is not None:
        print(f"  Output: {run.output_dir}")
