import tempfile
import unittest
from pathlib import Path

import numpy as np

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    camera_intrinsics,
    voxel_downsample,
    write_ply,
)


class GeometryCheckerTests(unittest.TestCase):
    def test_backprojection_uses_mujoco_camera_axes(self):
        depth = np.full((2, 2), 2.0, dtype=np.float32)
        mask = np.array([[True, False], [False, False]])
        intrinsics = np.array(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
        )
        points, pixels = backproject_masked_depth(
            depth,
            mask,
            intrinsics,
            camera_position=np.array([10.0, 20.0, 30.0]),
            camera_rotation=np.eye(3),
            max_depth=5.0,
        )
        np.testing.assert_allclose(points, [[9.0, 21.0, 28.0]])
        np.testing.assert_array_equal(pixels, [[0, 0]])

    def test_vertical_fov_intrinsics_are_square_pixel(self):
        intrinsics = camera_intrinsics(90.0, width=640, height=480)
        self.assertAlmostEqual(intrinsics[0, 0], 240.0)
        self.assertAlmostEqual(intrinsics[1, 1], 240.0)
        self.assertAlmostEqual(intrinsics[0, 2], 320.0)
        self.assertAlmostEqual(intrinsics[1, 2], 240.0)

    def test_voxel_fusion_removes_duplicate_samples(self):
        points = np.array(
            [[0.001, 0.001, 0.001], [0.002, 0.002, 0.002], [0.02, 0.0, 0.0]],
            dtype=np.float32,
        )
        colors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        fused_points, fused_colors = voxel_downsample(points, colors, 0.01)
        np.testing.assert_array_equal(fused_points, points[[0, 2]])
        np.testing.assert_array_equal(fused_colors, colors[[0, 2]])

    def test_binary_ply_contains_expected_vertex_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            write_ply(
                path,
                np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.array([[10, 20, 30]], dtype=np.uint8),
            )
            payload = path.read_bytes()
        self.assertIn(b"format binary_little_endian 1.0", payload)
        self.assertIn(b"element vertex 1", payload)


if __name__ == "__main__":
    unittest.main()
