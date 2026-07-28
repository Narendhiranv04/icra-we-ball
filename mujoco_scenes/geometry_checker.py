"""Five-view RGB-D object point-cloud reconstruction for MuJoCo scenes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import mujoco
except ModuleNotFoundError:  # Allow the pure geometry helpers to be unit tested.
    mujoco = None


DEFAULT_FUSION_CAMERAS = (
    "left_shoulder_camera",
    "right_shoulder_camera",
    "overhead_camera",
    "side_camera",
    "front_camera",
)


@dataclass
class ObjectPointCloud:
    """Fused world-frame samples belonging to one MuJoCo object body."""

    instance_name: str
    object_kind: str
    points: np.ndarray
    colors: np.ndarray
    pixels_by_camera: dict[str, int] = field(default_factory=dict)


@dataclass
class PointCloudRun:
    """Outputs and wall-clock timings from one five-view reconstruction."""

    clouds: dict[str, ObjectPointCloud]
    cameras: tuple[str, ...]
    width: int
    height: int
    timings_seconds: dict[str, float]
    output_dir: Path | None = None

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


def backproject_masked_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    *,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project selected pixels and return world points plus valid pixel indices.

    MuJoCo cameras look along local -Z with +X right and +Y up. Image row
    coordinates point downward, hence the sign on local Y.
    """
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
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
        cameras: Iterable[str] = DEFAULT_FUSION_CAMERAS,
        width: int = 640,
        height: int = 480,
        max_depth: float = 5.0,
        voxel_size: float = 0.003,
    ):
        if mujoco is None:
            raise RuntimeError("GeometryChecker requires the mujoco package")
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.cameras = tuple(cameras)
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.voxel_size = voxel_size
        self._camera_ids = {
            name: self._require_id(mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in self.cameras
        }

    def _require_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _geom_ids_by_instance(
        self, instance_names: Iterable[str]
    ) -> dict[str, np.ndarray]:
        root_by_body: dict[int, str] = {}
        for name in instance_names:
            body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, name)
            root_by_body[body_id] = name

        geom_ids: dict[str, list[int]] = {name: [] for name in instance_names}
        for geom_id, geom_body_id in enumerate(self.model.geom_bodyid):
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
        visible = self.scene.get_visible_object_instances()
        instance_kinds = {name: kind for name, kind in visible}
        geom_ids = self._geom_ids_by_instance(instance_kinds)
        instance_bounds = {}
        for instance_name, ids in geom_ids.items():
            body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, instance_name)
            body_position = self.data.xpos[body_id].copy()
            radius = max(
                (
                    np.linalg.norm(self.data.geom_xpos[geom_id] - body_position)
                    + float(self.model.geom_rbound[geom_id])
                    for geom_id in ids
                ),
                default=0.0,
            )
            instance_bounds[instance_name] = (body_position, radius)
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
        try:
            for camera_name in self.cameras:
                camera_id = self._camera_ids[camera_name]
                renderer.update_scene(self.data, camera=camera_id)

                render_started = time.perf_counter()
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
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
                    mask = is_geom & np.isin(segmentation[:, :, 0], ids)
                    world_points, pixels = backproject_masked_depth(
                        depth,
                        mask,
                        intrinsics,
                        camera_position,
                        camera_rotation,
                        max_depth=self.max_depth,
                    )
                    # Very small geoms can disagree by a boundary pixel between
                    # MuJoCo's ID-color and depth passes. Reject such background
                    # depths using the instance's live model-space bounding sphere.
                    body_position, radius = instance_bounds[instance_name]
                    inside = (
                        np.linalg.norm(world_points - body_position, axis=1)
                        <= radius + 0.02
                    )
                    world_points = world_points[inside]
                    pixels = pixels[inside]
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
