import unittest

from mujoco_scenes.place_motion import (
    DRAWER_REGIONS,
    SERVING_REGION,
    TABLE_SUBREGIONS,
    buffered_center_bounds,
    resolve_place_region,
)


class PlaceMotionTests(unittest.TestCase):
    def test_public_table_alias_resolves_by_robot_pose(self):
        self.assertEqual(resolve_place_region("table", "home").name, "table_sub_1")
        self.assertEqual(
            resolve_place_region("table", "cupboard1").name, "table_sub_2"
        )
        self.assertEqual(
            resolve_place_region("table", "right_side").name, "table_sub_3"
        )

    def test_serving_table_is_only_reachable_from_home(self):
        self.assertIs(resolve_place_region("serving_table", "home"), SERVING_REGION)
        with self.assertRaises(RuntimeError):
            resolve_place_region("serving_table", "cupboard1")

    def test_edge_buffer_and_object_footprint_shrink_sampling_rectangle(self):
        bounds = buffered_center_bounds(
            SERVING_REGION,
            (-0.05, 0.05, -0.04, 0.04),
        )
        self.assertAlmostEqual(bounds[0], -0.175)
        self.assertAlmostEqual(bounds[1], 0.175)
        self.assertAlmostEqual(bounds[2], -0.645)
        self.assertAlmostEqual(bounds[3], -0.475)

    def test_table_subregions_are_distinct_and_inside_countertop(self):
        self.assertEqual(len({region.name for region in TABLE_SUBREGIONS.values()}), 3)
        for region in TABLE_SUBREGIONS.values():
            min_x, max_x, min_y, max_y = region.bounds
            self.assertGreaterEqual(min_x, -0.70)
            self.assertLessEqual(max_x, 0.70)
            self.assertGreaterEqual(min_y, -0.40)
            self.assertLessEqual(max_y, 0.40)

    def test_drawer_regions_are_home_only_and_mirrored(self):
        d1 = resolve_place_region("drawer_D1", "home")
        d2 = resolve_place_region("drawer_D2", "home")
        self.assertIs(d1, DRAWER_REGIONS["drawer_D1"])
        self.assertIs(d2, DRAWER_REGIONS["drawer_D2"])
        self.assertAlmostEqual(d1.bounds[0], -d2.bounds[1])
        self.assertAlmostEqual(d1.bounds[1], -d2.bounds[0])
        self.assertEqual(d1.surface_geom, "D1_tray_base")
        self.assertEqual(d2.surface_geom, "D2_tray_base")
        with self.assertRaises(RuntimeError):
            resolve_place_region("drawer_D1", "right_side")


if __name__ == "__main__":
    unittest.main()
