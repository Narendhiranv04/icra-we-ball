import unittest

from mujoco_scenes.scene_loader import (
    ROBOT_FETCH,
    ROBOT_GOOGLE,
    ROBOT_NONE,
    ROBOT_CHOICES,
    _selected_robot,
)


class RobotSelectionTests(unittest.TestCase):
    def test_legacy_boolean_defaults_to_google_or_none(self):
        self.assertEqual(_selected_robot(True, None), ROBOT_GOOGLE)
        self.assertEqual(_selected_robot(False, None), ROBOT_NONE)

    def test_explicit_google_overrides_legacy_boolean(self):
        self.assertEqual(_selected_robot(True, ROBOT_GOOGLE), ROBOT_GOOGLE)
        self.assertEqual(_selected_robot(False, ROBOT_GOOGLE), ROBOT_GOOGLE)

    def test_fetch_backend_is_no_longer_selectable(self):
        with self.assertRaisesRegex(ValueError, "Unknown robot"):
            _selected_robot(True, ROBOT_FETCH)

    def test_unknown_robot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown robot"):
            _selected_robot(True, "unknown")

    def test_production_cli_exposes_google_and_no_robot_only(self):
        self.assertEqual(ROBOT_CHOICES, (ROBOT_GOOGLE, ROBOT_NONE))
        self.assertNotIn(ROBOT_FETCH, ROBOT_CHOICES)


if __name__ == "__main__":
    unittest.main()
