import unittest

import numpy as np

from mujoco_scenes.drawer_motion import (
    DRAWER_FRONT_GRASP_ROTATION,
    DRAWER_PREGRASP_DISTANCE,
    DRAWER_PULL_SAMPLES,
    DRAWER_RETREAT_DISTANCE,
    DRAWER_SPECS,
)


class DrawerMotionTests(unittest.TestCase):
    def test_both_drawers_use_matching_physical_elements(self):
        for index, name in enumerate(("D1", "D2"), start=1):
            spec = DRAWER_SPECS[name]
            self.assertEqual(spec.label, f"Drawer {index}")
            self.assertEqual(spec.joint, f"{name}_slide_joint")
            self.assertEqual(
                spec.handle_geoms,
                frozenset(
                    {
                        f"{name}_handle_left",
                        f"{name}_handle_right",
                        f"{name}_handle_bar",
                    }
                ),
            )

    def test_front_grasp_approaches_positive_y_with_vertical_fingers(self):
        rotation = DRAWER_FRONT_GRASP_ROTATION
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rotation[:, 0], (0.0, 1.0, 0.0))
        np.testing.assert_allclose(rotation[:, 1], (0.0, 0.0, 1.0))

    def test_hover_pull_and_horizontal_retreat_are_explicit(self):
        self.assertAlmostEqual(DRAWER_PREGRASP_DISTANCE, 0.080)
        self.assertAlmostEqual(DRAWER_RETREAT_DISTANCE, 0.200)
        self.assertGreater(DRAWER_PULL_SAMPLES, 20)


if __name__ == "__main__":
    unittest.main()
