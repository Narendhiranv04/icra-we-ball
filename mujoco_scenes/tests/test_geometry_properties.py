import unittest

import numpy as np

from mujoco_scenes.geometry_properties import (
    extract_object_properties,
    load_semantics_config,
    pairwise_relation_status,
)


class GeometryPropertyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_semantics_config()

    def test_robust_pca_ignores_nonfinite_points_and_outliers(self):
        rng = np.random.default_rng(4)
        points = rng.uniform(
            low=(-0.10, -0.02, -0.01),
            high=(0.10, 0.02, 0.01),
            size=(1000, 3),
        )
        points = np.vstack((points, [np.nan, 0, 0], [50, 50, 50]))
        properties = extract_object_properties(
            points,
            category="spoon",
            contributing_camera_count=3,
            config=self.config,
        )
        dimensions = properties["dimensions_m"]
        self.assertEqual(properties["property_status"], "MEASURED")
        self.assertAlmostEqual(dimensions["length"]["value"], 0.192, delta=0.012)
        self.assertLess(dimensions["width"]["value"], 0.05)
        self.assertEqual(dimensions["length"]["method"], "robust_pca_obb")

    def test_family_members_have_identical_property_keys(self):
        points = np.linspace((0, 0, 0), (0.2, 0.02, 0.01), 100)
        spoon = extract_object_properties(
            points,
            category="spoon",
            contributing_camera_count=2,
            config=self.config,
        )
        fork = extract_object_properties(
            points,
            category="fork",
            contributing_camera_count=2,
            config=self.config,
        )
        self.assertEqual(
            set(spoon["family_properties"]),
            {"total_length_m", "maximum_width_m"},
        )
        self.assertEqual(
            set(spoon["family_properties"]),
            set(fork["family_properties"]),
        )

    def test_invalid_or_insufficient_cloud_is_unknown(self):
        properties = extract_object_properties(
            np.array([[np.nan, 0, 0], [0, np.inf, 0], [0, 0, 0]]),
            category="mug",
            contributing_camera_count=0,
            config=self.config,
        )
        self.assertEqual(properties["property_status"], "UNKNOWN")
        self.assertIsNone(properties["centroid_world_m"]["value"])
        self.assertTrue(
            all(
                record["status"] == "UNKNOWN" and record["value"] is None
                for record in properties["family_properties"].values()
            )
        )

    def test_receptacle_estimator_uses_visible_rim_and_interior(self):
        angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        rim = np.column_stack(
            (0.04 * np.cos(angles), 0.04 * np.sin(angles), np.full(96, 0.10))
        )
        side = np.column_stack(
            (
                0.04 * np.cos(np.repeat(angles, 2)),
                0.04 * np.sin(np.repeat(angles, 2)),
                np.tile((0.03, 0.07), len(angles)),
            )
        )
        interior = np.column_stack(
            (
                np.linspace(-0.01, 0.01, 25),
                np.zeros(25),
                np.full(25, 0.02),
            )
        )
        properties = extract_object_properties(
            np.vstack((rim, side, interior)),
            category="mug",
            contributing_camera_count=4,
            config=self.config,
        )
        family = properties["family_properties"]
        self.assertEqual(family["opening_width_m"]["status"], "MEASURED")
        self.assertEqual(family["cavity_depth_m"]["status"], "MEASURED")

    def test_pairwise_relation_is_unknown_when_measurement_is_missing(self):
        utensil = extract_object_properties(
            np.linspace((0, 0, 0), (0.2, 0.02, 0.01), 100),
            category="spoon",
            contributing_camera_count=2,
            config=self.config,
        )
        receptacle = extract_object_properties(
            np.zeros((2, 3)),
            category="mug",
            contributing_camera_count=0,
            config=self.config,
        )
        self.assertEqual(
            pairwise_relation_status(
                "INSERTABLE_IN", utensil, receptacle, self.config
            ),
            "UNKNOWN",
        )
        self.assertEqual(
            pairwise_relation_status(
                "REACHES_BOTTOM", utensil, receptacle, self.config
            ),
            "UNKNOWN",
        )


if __name__ == "__main__":
    unittest.main()
