import unittest
from unittest import mock

import numpy as np

from mujoco_scenes.scene_loader import (
    KitchenScene,
    ROBOT_GOOGLE,
    ROBOT_NONE,
    ROBOT_CHOICES,
    _selected_robot,
    _validate_step_count,
)


class RobotSelectionTests(unittest.TestCase):
    def test_legacy_boolean_defaults_to_google_or_none(self):
        self.assertEqual(_selected_robot(True, None), ROBOT_GOOGLE)
        self.assertEqual(_selected_robot(False, None), ROBOT_NONE)

    def test_explicit_google_overrides_legacy_boolean(self):
        self.assertEqual(_selected_robot(True, ROBOT_GOOGLE), ROBOT_GOOGLE)
        self.assertEqual(_selected_robot(False, ROBOT_GOOGLE), ROBOT_GOOGLE)

    def test_unknown_robot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown robot"):
            _selected_robot(True, "unknown")

    def test_production_cli_exposes_google_and_no_robot_only(self):
        self.assertEqual(ROBOT_CHOICES, (ROBOT_GOOGLE, ROBOT_NONE))


class SceneBoundaryTests(unittest.TestCase):
    def test_step_count_must_be_a_positive_integer(self):
        self.assertEqual(_validate_step_count(1), 1)
        for invalid in (0, -1, True, 1.5, "10"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _validate_step_count(invalid)

    @mock.patch("mujoco_scenes.scene_loader.mujoco.mj_forward")
    @mock.patch("mujoco_scenes.scene_loader.mujoco.mj_name2id", return_value=2)
    @mock.patch("mujoco_scenes.scene_loader.mujoco.Renderer")
    def test_render_frame_closes_renderer(
        self, renderer_class, _name_to_id, _forward
    ):
        renderer = renderer_class.return_value
        renderer.render.return_value = np.zeros((3, 4, 3), dtype=np.uint8)
        scene = KitchenScene.__new__(KitchenScene)
        scene.model = object()
        scene.data = object()

        frame = scene.render_frame("front_camera", width=4, height=3)

        self.assertEqual(frame.shape, (3, 4, 3))
        renderer.close.assert_called_once_with()

    @mock.patch("mujoco_scenes.scene_loader.mujoco.mj_name2id", return_value=-1)
    def test_render_frame_rejects_unknown_camera(self, _name_to_id):
        scene = KitchenScene.__new__(KitchenScene)
        scene.model = object()
        scene.data = object()
        with self.assertRaisesRegex(ValueError, "Unknown fixed camera"):
            scene.render_frame("missing", width=4, height=3)

    def test_render_frame_rejects_invalid_dimensions_before_allocation(self):
        scene = KitchenScene.__new__(KitchenScene)
        for width, height in ((0, 3), (4, -1), (True, 3), (4.0, 3)):
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    scene.render_frame("front_camera", width=width, height=height)


if __name__ == "__main__":
    unittest.main()
