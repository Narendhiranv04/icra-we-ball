import unittest

import numpy as np

from mujoco_scenes.generic_manipulation import (
    ARM_COMMAND_SPEED,
    BASE_LINEAR_COMMAND_SPEED as MANIPULATION_BASE_LINEAR_COMMAND_SPEED,
    BASE_YAW_COMMAND_SPEED as MANIPULATION_BASE_YAW_COMMAND_SPEED,
    CALIBRATED_SCENE_OBJECTS,
    GOOGLE_CALIBRATION_PICK_SPECS,
    GOOGLE_PICK_SPECS,
    INTERMEDIATE_TRACKING_TOLERANCE,
    JOINT_WAYPOINT_TOLERANCE,
    MANIPULATION_BASE_LINEAR_DAMPING,
    SELF_COLLISION_MOUNT_ALLOWANCES,
    SPOON_PIVOT_RELAXATION,
    SPOON_REGRASP_SQUEEZE,
    WAYPOINT_HOLD_TICKS,
)
from mujoco_scenes.mobile_motion import (
    BASE_LINEAR_COMMAND_SPEED as NAVIGATION_BASE_LINEAR_COMMAND_SPEED,
    BASE_YAW_COMMAND_SPEED as NAVIGATION_BASE_YAW_COMMAND_SPEED,
)
from mujoco_scenes.robot_profiles import (
    GOOGLE_LEFT_FINGER_GEOMS,
    GOOGLE_RIGHT_FINGER_GEOMS,
    manipulation_profile,
    mobile_profile,
)


class RobotProfileTests(unittest.TestCase):
    def test_google_mobile_profile_uses_namespaced_planar_controls(self):
        profile = mobile_profile("google")
        self.assertEqual(profile.body_prefix, "google:")
        self.assertEqual(profile.base_joints[0], "google:base_forward_joint")
        self.assertEqual(profile.base_actuators[2], "google:base_yaw_actuator")
        self.assertEqual(profile.home_y, -1.25)
        self.assertGreaterEqual(profile.forward_limits[1], 1.25)

    def test_google_top_down_frame_is_right_handed_and_points_local_z_down(self):
        rotation = manipulation_profile("google").top_down_rotation
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        np.testing.assert_allclose(rotation[:, 2], (0.0, 0.0, -1.0))

    def test_only_the_physical_shoulder_mount_has_a_self_overlap_allowance(self):
        self.assertEqual(
            SELF_COLLISION_MOUNT_ALLOWANCES,
            {
                frozenset(("google:base_link", "google:link_shoulder")): -0.100,
                frozenset(("google:base_link", "google:link_bicep")): -0.050,
            },
        )

    def test_google_gripper_closes_by_increasing_angular_commands(self):
        profile = manipulation_profile("google")
        self.assertLess(profile.open_command, profile.closed_command)
        self.assertEqual(profile.open_command, 0.01)
        self.assertEqual(profile.closed_command, 1.30)
        self.assertEqual(profile.close_step, 0.003)
        np.testing.assert_allclose(profile.navigation_joints, 0.0)

    def test_google_motion_uses_bounded_consistent_command_rates(self):
        self.assertEqual(ARM_COMMAND_SPEED, 1.20)
        self.assertEqual(MANIPULATION_BASE_LINEAR_COMMAND_SPEED, 0.25)
        self.assertEqual(MANIPULATION_BASE_YAW_COMMAND_SPEED, 0.60)
        self.assertEqual(
            MANIPULATION_BASE_LINEAR_COMMAND_SPEED,
            NAVIGATION_BASE_LINEAR_COMMAND_SPEED,
        )
        self.assertEqual(
            MANIPULATION_BASE_YAW_COMMAND_SPEED,
            NAVIGATION_BASE_YAW_COMMAND_SPEED,
        )
        self.assertGreater(INTERMEDIATE_TRACKING_TOLERANCE, JOINT_WAYPOINT_TOLERANCE)
        self.assertEqual(WAYPOINT_HOLD_TICKS, 4)
        self.assertEqual(MANIPULATION_BASE_LINEAR_DAMPING, 2000.0)

    def test_google_requires_named_contacts_from_both_fingers(self):
        profile = manipulation_profile("google")
        self.assertEqual(
            profile.finger_contact_geoms,
            (GOOGLE_RIGHT_FINGER_GEOMS, GOOGLE_LEFT_FINGER_GEOMS),
        )
        self.assertEqual(len(GOOGLE_RIGHT_FINGER_GEOMS), 6)
        self.assertEqual(len(GOOGLE_LEFT_FINGER_GEOMS), 6)
        self.assertTrue(GOOGLE_RIGHT_FINGER_GEOMS.isdisjoint(GOOGLE_LEFT_FINGER_GEOMS))

    def test_only_physically_validated_google_object_is_exposed(self):
        profile = manipulation_profile("google")
        self.assertEqual(profile.supported_objects, ("sugar_jar", "spoon"))
        self.assertIn("sugar_jar", GOOGLE_PICK_SPECS)
        spoon = GOOGLE_PICK_SPECS["spoon"]
        self.assertEqual(spoon.required_contact_geoms, ("spoon_handle_collision",))
        self.assertFalse(spoon.place_supported)
        self.assertEqual(SPOON_PIVOT_RELAXATION, 0.30)
        self.assertGreater(SPOON_REGRASP_SQUEEZE, 0.0)
        self.assertLess(SPOON_REGRASP_SQUEEZE, SPOON_PIVOT_RELAXATION)
        np.testing.assert_allclose(spoon.top_down_rotation[:, 1], (0.0, -1.0, 0.0))
        self.assertLess(spoon.carry_position[1], profile.carry_position[1])
        self.assertEqual(
            CALIBRATED_SCENE_OBJECTS,
            {"S1_coffee_missing_mug": ("sugar_jar", "spoon")},
        )

    def test_uncalibrated_google_candidates_are_declared_separately(self):
        self.assertEqual(
            set(GOOGLE_CALIBRATION_PICK_SPECS), {"coffee_jar", "kettle"}
        )
        self.assertTrue(
            set(GOOGLE_CALIBRATION_PICK_SPECS).isdisjoint(
                manipulation_profile("google").supported_objects
            )
        )
        self.assertFalse(
            GOOGLE_CALIBRATION_PICK_SPECS["coffee_jar"].place_supported
        )
        kettle = GOOGLE_CALIBRATION_PICK_SPECS["kettle"]
        self.assertEqual(
            kettle.required_contact_geoms, ("kettle_handle_collision",)
        )
        self.assertFalse(kettle.place_supported)

    def test_unknown_profiles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no mobile-motion profile"):
            mobile_profile("unknown")
        with self.assertRaisesRegex(ValueError, "no generic manipulation profile"):
            manipulation_profile("unknown")


if __name__ == "__main__":
    unittest.main()
