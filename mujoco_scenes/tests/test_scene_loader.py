import unittest

from mujoco_scenes.scene_loader import (
    ROBOT_FETCH,
    ROBOT_GOOGLE,
    ROBOT_NONE,
    _selected_robot,
)


class RobotSelectionTests(unittest.TestCase):
    def test_legacy_boolean_defaults_to_fetch_or_none(self):
        self.assertEqual(_selected_robot(True, None), ROBOT_FETCH)
        self.assertEqual(_selected_robot(False, None), ROBOT_NONE)

    def test_explicit_robot_overrides_legacy_boolean(self):
        self.assertEqual(_selected_robot(True, ROBOT_GOOGLE), ROBOT_GOOGLE)
        self.assertEqual(_selected_robot(False, ROBOT_FETCH), ROBOT_FETCH)

    def test_unknown_robot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown robot"):
            _selected_robot(True, "unknown")


if __name__ == "__main__":
    unittest.main()
