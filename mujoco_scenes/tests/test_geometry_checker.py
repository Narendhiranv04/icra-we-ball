import tempfile
import unittest
from pathlib import Path

import numpy as np

from mujoco_scenes.geometry_checker import (
    associate_segmented_centroid,
    backproject_masked_depth,
    camera_intrinsics,
    gate_points_to_volume,
    load_inspection_rig_config,
    look_at_camera_rotation,
    scale_intrinsics,
    voxel_downsample,
    write_ply,
)
from mujoco_scenes.living_room_scene import LIVING_ROOM_INSPECTION_RIG_CONFIG


class GeometryCheckerTests(unittest.TestCase):
    def test_learned_masks_associate_by_label_and_world_centroid(self):
        centroids = {}
        labels = {}
        observations = {}
        first = associate_segmented_centroid(
            "spoon",
            np.array([0.1, 0.2, 0.3]),
            track_centroids=centroids,
            track_labels=labels,
            track_observations=observations,
            used_track_ids=set(),
            maximum_distance_m=0.12,
        )
        second = associate_segmented_centroid(
            "spoon",
            np.array([0.12, 0.2, 0.3]),
            track_centroids=centroids,
            track_labels=labels,
            track_observations=observations,
            used_track_ids=set(),
            maximum_distance_m=0.12,
        )
        third = associate_segmented_centroid(
            "spoon",
            np.array([0.5, 0.2, 0.3]),
            track_centroids=centroids,
            track_labels=labels,
            track_observations=observations,
            used_track_ids=set(),
            maximum_distance_m=0.12,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

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

    def test_intrinsics_scale_correctly_to_320_by_240(self):
        source = camera_intrinsics(60.0, width=640, height=480)
        scaled = scale_intrinsics(
            source,
            source_width=640,
            source_height=480,
            target_width=320,
            target_height=240,
        )
        direct = camera_intrinsics(60.0, width=320, height=240)
        np.testing.assert_allclose(scaled, direct)

    def test_known_multiview_camera_transforms_align(self):
        world_point = np.array([0.1, 0.2, 0.6])
        intrinsics = np.array(
            [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
        )
        reconstructed = []
        for position in (
            np.array([0.1, -0.8, 0.6]),
            np.array([-0.9, 0.2, 0.6]),
        ):
            rotation = look_at_camera_rotation(
                position, world_point, np.array([0.0, 0.0, 1.0])
            )
            local = rotation.T @ (world_point - position)
            depth = -local[2]
            column = intrinsics[0, 0] * local[0] / depth + intrinsics[0, 2]
            row = intrinsics[1, 2] - intrinsics[1, 1] * local[1] / depth
            self.assertAlmostEqual(column, 1.0)
            self.assertAlmostEqual(row, 1.0)
            depth_image = np.full((3, 3), np.nan, dtype=np.float32)
            depth_image[1, 1] = depth
            mask = np.zeros((3, 3), dtype=bool)
            mask[1, 1] = True
            points, _pixels = backproject_masked_depth(
                depth_image,
                mask,
                intrinsics,
                position,
                rotation,
                max_depth=2.0,
            )
            reconstructed.append(points[0])
        np.testing.assert_allclose(reconstructed, [world_point, world_point])

    def test_region_gate_rejects_tabletop_points_for_c2(self):
        points = np.array(
            [
                [0.35, 0.65, 0.95],
                [-0.15, -0.30, 0.68],
            ]
        )
        inside = gate_points_to_volume(
            points,
            minimum_world_m=np.array([0.12, 0.43, 0.72]),
            maximum_world_m=np.array([0.58, 0.84, 1.19]),
            boundary_margin_m=0.02,
        )
        np.testing.assert_array_equal(inside, [True, False])

    def test_every_region_has_five_distinct_facing_views(self):
        config = load_inspection_rig_config()
        self.assertEqual(
            set(config["regions"]),
            {"INITIAL", "D1", "D2", "C2", "B1", "C1"},
        )
        for region in config["regions"].values():
            self.assertEqual(len(region["cameras"]), 5)

    def test_living_room_uses_the_five_robot_top_cameras(self):
        config = load_inspection_rig_config(
            LIVING_ROOM_INSPECTION_RIG_CONFIG
        )
        self.assertEqual(
            config["inspection_sequence"],
            ["LEFT_DRAWER", "RIGHT_DRAWER"],
        )
        self.assertEqual(
            set(config["regions"]),
            {"INITIAL", "LEFT_DRAWER", "RIGHT_DRAWER"},
        )
        self.assertEqual(
            set(config["camera_slots"].values()),
            {
                "top_front_camera",
                "top_front_left_camera",
                "top_rear_left_camera",
                "top_rear_right_camera",
                "top_front_right_camera",
            },
        )
        for region in config["regions"].values():
            self.assertEqual(len(region["cameras"]), 5)

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
