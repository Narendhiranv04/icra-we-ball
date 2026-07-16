import unittest

import numpy as np

from mujoco_scenes.open_motion import (
    BOX_ARC_SAMPLES,
    BOX_GRASP_ROTATION,
    BOX_HANDLE_GEOMS,
    BOX_PREGRASP_DISTANCE,
)


class OpenMotionTests(unittest.TestCase):
    def test_box_grasp_is_horizontal_with_vertical_fingers(self):
        np.testing.assert_allclose(
            BOX_GRASP_ROTATION.T @ BOX_GRASP_ROTATION,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(BOX_GRASP_ROTATION[:, 0], (0.0, 1.0, 0.0))
        self.assertAlmostEqual(abs(BOX_GRASP_ROTATION[:, 1] @ (0.0, 0.0, 1.0)), 1.0)

    def test_box_uses_a_short_positive_y_insertion(self):
        self.assertAlmostEqual(BOX_PREGRASP_DISTANCE, 0.075)
        self.assertGreater(BOX_ARC_SAMPLES, 20)

    def test_only_physical_lid_handle_geometries_confirm_contact(self):
        self.assertEqual(
            BOX_HANDLE_GEOMS,
            {
                "B1_lid_handle_left",
                "B1_lid_handle_right",
                "B1_lid_handle_bar",
            },
        )


if __name__ == "__main__":
    unittest.main()
