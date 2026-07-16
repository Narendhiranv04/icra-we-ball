import math
import unittest

from mujoco_scenes.mobile_motion import (
    LEFT_POSE,
    RIGHT_POSE,
    RRTStarPlanner,
    anchor_route,
    physical_location,
)


class MobileMotionTests(unittest.TestCase):
    def test_cupboard2_and_box_are_physical_aliases(self):
        self.assertEqual(physical_location("cupboard2"), "right_side")
        self.assertEqual(physical_location("box"), "right_side")

    def test_side_poses_face_inward(self):
        self.assertAlmostEqual(LEFT_POSE.yaw, -math.pi / 2)
        self.assertAlmostEqual(RIGHT_POSE.yaw, math.pi / 2)

    def test_route_between_sides_returns_through_home(self):
        self.assertEqual(
            anchor_route("cupboard1", "right_side"),
            [
                "cupboard1",
                "left_clearance",
                "left_staging",
                "home",
                "right_staging",
                "right_clearance",
                "right_side",
            ],
        )

    def test_rrt_star_routes_around_obstacle(self):
        def state_valid(x, y):
            return not (-0.20 <= x <= 0.20 and -0.60 <= y <= 0.20)

        planner = RRTStarPlanner(
            state_valid,
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            max_iterations=2500,
            seed=4,
        )
        path = planner.plan((0.0, -0.9), (0.0, 0.6))
        self.assertGreater(len(path), 2)
        self.assertTrue(all(state_valid(x, y) for x, y in path))
        self.assertTrue(any(abs(x) > 0.20 for x, _ in path))


if __name__ == "__main__":
    unittest.main()
