import unittest
import math

import numpy as np

from mujoco_scenes.pick_motion import (
    APPROACH_CLEARANCE,
    CARRY_POSITION,
    HORIZONTAL_CARRY_POSITION,
    JAR_HORIZONTAL_ROTATION,
    JAR_TOP_DOWN_ROTATION,
    LIFT_CLEARANCE,
    DRAWER_PICK_SPECS,
    PICK_SPECS,
    SPOON_POST_GRASP_CLEARANCE,
    PickExecutor,
    TABLE_PICK_SPECS,
    TOP_DOWN_ROTATION,
    carry_position_at,
    top_down_rotation_for_body,
)


class PickMotionTests(unittest.TestCase):
    def test_table_objects_have_explicit_grasp_sites(self):
        self.assertEqual(
            set(TABLE_PICK_SPECS),
            {"kettle", "coffee_jar", "sugar_jar", "spoon"},
        )
        self.assertEqual(TABLE_PICK_SPECS["kettle"].grasp_site, "kettle_grasp")
        self.assertTrue(TABLE_PICK_SPECS["kettle"].align_to_body)
        self.assertEqual(
            TABLE_PICK_SPECS["kettle"].required_contact_geoms,
            ("kettle_handle_collision",),
        )
        self.assertEqual(TABLE_PICK_SPECS["spoon"].grasp_site, "spoon_grasp")
        self.assertTrue(TABLE_PICK_SPECS["coffee_jar"].reorient_horizontal)
        self.assertTrue(TABLE_PICK_SPECS["sugar_jar"].reorient_horizontal)

    def test_drawer_objects_use_end_hangs_except_centre_pinched_tissue(self):
        self.assertEqual(
            set(DRAWER_PICK_SPECS),
            {
                "fork",
                "knife",
                "stirrer",
                "spatula",
                "tongs",
                "napkin",
                "gso_spatula_distractor",
            },
        )
        for name, spec in DRAWER_PICK_SPECS.items():
            self.assertTrue(spec.align_to_body)
            if name == "napkin":
                self.assertFalse(spec.passive_hang)
                self.assertTrue(spec.centre_pinch_assist)
                self.assertEqual(spec.grasp_site, "napkin_grasp")
            else:
                self.assertTrue(spec.passive_hang)
                self.assertTrue(spec.handle_pinch_assist)
                self.assertEqual(spec.hang_axis_local, (1.0, 0.0, 0.0))
        self.assertEqual(
            set(PICK_SPECS), set(TABLE_PICK_SPECS) | set(DRAWER_PICK_SPECS)
        )

    def test_vertical_approach_and_lift_use_one_constant_distance(self):
        self.assertAlmostEqual(APPROACH_CLEARANCE, 0.08)
        self.assertEqual(APPROACH_CLEARANCE, LIFT_CLEARANCE)
        self.assertAlmostEqual(SPOON_POST_GRASP_CLEARANCE, 0.16)

    def test_carry_position_rotates_with_side_base_pose(self):
        right = carry_position_at(CARRY_POSITION, "right_side")
        np.testing.assert_allclose(right, (0.745, -0.10, 0.95), atol=1e-12)

    def test_top_down_rotation_is_orthonormal_and_points_down(self):
        np.testing.assert_allclose(
            TOP_DOWN_ROTATION.T @ TOP_DOWN_ROTATION, np.eye(3), atol=1e-12
        )
        np.testing.assert_allclose(TOP_DOWN_ROTATION[:, 0], (0.0, 0.0, -1.0))

    def test_kettle_alignment_tracks_body_yaw(self):
        yaw = math.radians(37.0)
        body_rotation = np.array(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        rotation = top_down_rotation_for_body(body_rotation)
        np.testing.assert_allclose(rotation[:, 0], (0.0, 0.0, -1.0))
        self.assertAlmostEqual(abs(rotation[:, 1] @ body_rotation[:, 1]), 1.0)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)

    def test_kettle_closes_across_the_diagonal_handle_tube(self):
        spec = TABLE_PICK_SPECS["kettle"]
        rotation = top_down_rotation_for_body(
            np.eye(3), spec.closing_axis_local
        )
        expected = np.array((math.sqrt(0.5), math.sqrt(0.5), 0.0))
        np.testing.assert_allclose(rotation[:, 1], expected, atol=1e-12)

    def test_jar_reorientation_is_a_90_degree_contact_axis_pitch(self):
        np.testing.assert_allclose(
            JAR_TOP_DOWN_ROTATION.T @ JAR_TOP_DOWN_ROTATION,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            JAR_HORIZONTAL_ROTATION.T @ JAR_HORIZONTAL_ROTATION,
            np.eye(3),
            atol=1e-12,
        )
        # Finger closing direction is invariant, while approach changes from
        # vertical to horizontal by exactly 90 degrees.
        np.testing.assert_allclose(
            JAR_TOP_DOWN_ROTATION[:, 1],
            JAR_HORIZONTAL_ROTATION[:, 1],
            atol=1e-12,
        )
        self.assertAlmostEqual(
            JAR_TOP_DOWN_ROTATION[:, 0] @ JAR_HORIZONTAL_ROTATION[:, 0],
            0.0,
        )
        self.assertAlmostEqual(HORIZONTAL_CARRY_POSITION[2], 0.74)

    def test_horizontal_jar_approach_stays_horizontal_at_side_yaws(self):
        for yaw in (-math.pi / 2, math.pi / 2):
            cosine, sine = math.cos(yaw), math.sin(yaw)
            base_rotation = np.array(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
            )
            rotated = base_rotation @ JAR_HORIZONTAL_ROTATION
            self.assertAlmostEqual(float(rotated[:, 0] @ (0.0, 0.0, 1.0)), 0.0)

    def test_joint_path_has_continuous_nonzero_internal_velocity(self):
        points = np.repeat(np.array(((0.0,), (0.2,), (0.4,), (0.6,))), 7, axis=1)
        times = np.array((0.0, 1.0, 2.0, 3.0))
        derivatives = PickExecutor._pchip_derivatives(points, times)
        np.testing.assert_allclose(derivatives[0], 0.0)
        np.testing.assert_allclose(derivatives[-1], 0.0)
        self.assertTrue(np.all(derivatives[1:-1] > 0.0))

    def test_joint_path_stops_only_at_a_true_geometric_reversal(self):
        points = np.array(((0.0, 0.0), (0.2, 0.1), (0.0, 0.0)))
        derivatives = PickExecutor._pchip_derivatives(
            points, np.array((0.0, 1.0, 2.0))
        )
        np.testing.assert_allclose(derivatives[1], 0.0)

    def test_tiny_ik_reversal_does_not_force_an_artificial_stop(self):
        points = np.array(((0.0,), (0.10,), (0.099,)))
        derivatives = PickExecutor._pchip_derivatives(
            points, np.array((0.0, 1.0, 2.0))
        )
        self.assertNotEqual(float(derivatives[1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
