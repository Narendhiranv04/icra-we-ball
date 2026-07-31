import unittest

import numpy as np

from mujoco_scenes.geometry_checker import (
    CUMULATIVE_VISUALIZATION_PURPOSE,
    MeasurementEvidence,
)
from mujoco_scenes.geometry_properties import (
    GEOMETRIC_PREDICATE_KEYS,
    GEOMETRIC_PROPERTY_KEYS,
    extract_object_properties,
    load_geometry_config,
    pairwise_relation_evaluation,
    pairwise_relation_status,
)


def cavity_cloud(include_interior=True):
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    rim = np.column_stack(
        (
            0.05 * np.cos(angles),
            0.04 * np.sin(angles),
            np.full(len(angles), 0.10),
        )
    )
    heights = np.linspace(0.02, 0.09, 5)
    wall_angles, wall_heights = np.meshgrid(angles, heights)
    side = np.column_stack(
        (
            0.05 * np.cos(wall_angles.ravel()),
            0.04 * np.sin(wall_angles.ravel()),
            wall_heights.ravel(),
        )
    )
    if not include_interior:
        return np.vstack((rim, side))
    x, y = np.meshgrid(
        np.linspace(-0.018, 0.018, 12),
        np.linspace(-0.012, 0.012, 10),
    )
    interior = np.column_stack(
        (x.ravel(), y.ravel(), np.full(x.size, 0.015))
    )
    return np.vstack((rim, side, interior))


def evidence(
    points,
    *,
    cameras=("inspection_left", "inspection_right", "inspection_top"),
    quality_valid=True,
    path="stages/003_after_C2/evidence/object_0001/fused.ply",
    source_stage=3,
    source_region="C2",
    purpose="MEASUREMENT_EVIDENCE",
):
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    colors = np.tile(np.array([[80, 140, 210]], dtype=np.uint8), (len(points), 1))
    return MeasurementEvidence(
        instance_name="synthetic_instance",
        measurement_points=points,
        measurement_colors=colors,
        contributing_camera_ids=tuple(cameras),
        points_by_camera={camera: points.copy() for camera in cameras},
        source_stage=source_stage,
        source_region=source_region,
        measurement_cloud_path=path,
        measurement_quality={
            "quality_is_valid": quality_valid,
            "status": "VALID" if quality_valid else "INVALID",
            "reasons": [] if quality_valid else ["INSUFFICIENT_CAMERA_COVERAGE"],
            "point_count": len(points),
            "raw_inside_point_count": len(points),
            "outlier_points_removed": 0,
            "contributing_camera_count": len(cameras),
        },
        cloud_purpose=purpose,
    )


class GeometryPropertyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_geometry_config()

    def measure(self, points, **kwargs):
        return extract_object_properties(
            evidence(points, **kwargs),
            config=self.config,
        )

    def test_robust_pca_ignores_nonfinite_points_and_outliers(self):
        rng = np.random.default_rng(4)
        points = rng.uniform(
            low=(-0.10, -0.02, -0.01),
            high=(0.10, 0.02, 0.01),
            size=(1000, 3),
        )
        points = np.vstack((points, [np.nan, 0, 0], [50, 50, 50]))
        properties = self.measure(points)
        dimensions = properties["dimensions_m"]
        self.assertEqual(properties["property_status"], "MEASURED")
        self.assertAlmostEqual(
            dimensions["length"]["value"], 0.192, delta=0.012
        )
        self.assertLess(dimensions["width"]["value"], 0.05)

    def test_every_object_has_identical_universal_keys(self):
        points = np.linspace((0, 0, 0), (0.2, 0.02, 0.01), 100)
        first = self.measure(points)
        second = self.measure(points)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first["geometric_properties"]), set(GEOMETRIC_PROPERTY_KEYS)
        )
        self.assertEqual(
            set(first["geometric_predicates"]),
            set(GEOMETRIC_PREDICATE_KEYS),
        )

    def test_extractor_rejects_raw_arrays_and_historical_clouds(self):
        with self.assertRaises(TypeError):
            extract_object_properties(
                np.zeros((20, 3)), config=self.config
            )
        for path in (
            "objects/object_0001/cumulative.ply",
            "objects/object_0001/cumulative_visualization.ply",
            "stages/003/combined_cloud.ply",
        ):
            with self.assertRaises(ValueError):
                extract_object_properties(
                    evidence(cavity_cloud(), path=path),
                    config=self.config,
                )
        with self.assertRaises(ValueError):
            extract_object_properties(
                evidence(
                    cavity_cloud(),
                    purpose=CUMULATIVE_VISUALIZATION_PURPOSE,
                ),
                config=self.config,
            )

    def test_measurement_evidence_cannot_carry_semantic_or_asset_labels(self):
        fields = set(MeasurementEvidence.__dataclass_fields__)
        self.assertTrue(
            fields.isdisjoint(
                {
                    "category",
                    "canonical_label",
                    "semantic_label",
                    "body_name",
                    "geom_name",
                    "mesh_name",
                    "asset_name",
                }
            )
        )

    def test_invalid_measurement_quality_produces_unknown(self):
        properties = self.measure(
            cavity_cloud(),
            cameras=("inspection_left",),
            quality_valid=False,
        )
        self.assertEqual(properties["property_status"], "UNKNOWN")
        self.assertTrue(
            all(
                record["status"] == "UNKNOWN"
                for record in properties["geometric_properties"].values()
            )
        )
        self.assertTrue(
            all(
                record["status"] == "UNKNOWN"
                for record in properties["geometric_predicates"].values()
            )
        )

    def test_insufficient_point_count_is_unknown(self):
        properties = self.measure(np.zeros((10, 3)))
        predicate = properties["geometric_predicates"]["OPEN_CAVITY"]
        self.assertEqual(predicate["status"], "UNKNOWN")
        self.assertEqual(predicate["reason"], "INSUFFICIENT_POINT_COUNT")

    def test_open_cavity_requires_rim_and_multiview_interior(self):
        properties = self.measure(cavity_cloud())
        predicate = properties["geometric_predicates"]["OPEN_CAVITY"]
        self.assertEqual(predicate["status"], "TRUE")
        self.assertEqual(predicate["evidence"]["angular_bin_count"], 12)
        self.assertGreaterEqual(predicate["evidence"]["interior_camera_count"], 2)
        geometry = properties["geometric_properties"]
        self.assertEqual(geometry["opening_width_m"]["status"], "MEASURED")
        self.assertEqual(geometry["cavity_depth_m"]["status"], "MEASURED")

    def test_missing_central_depth_without_interior_is_unknown(self):
        properties = self.measure(cavity_cloud(include_interior=False))
        predicate = properties["geometric_predicates"]["OPEN_CAVITY"]
        self.assertEqual(predicate["status"], "UNKNOWN")
        self.assertEqual(predicate["reason"], "INSUFFICIENT_OBSERVED_INTERIOR")

    def test_flat_plate_and_segmentation_hole_are_not_open_cavities(self):
        x, y = np.meshgrid(
            np.linspace(-0.10, 0.10, 35),
            np.linspace(-0.07, 0.07, 28),
        )
        flat = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        plate = self.measure(flat)
        self.assertEqual(
            plate["geometric_predicates"]["OPEN_CAVITY"]["status"],
            "FALSE",
        )
        radius = np.sqrt(x.ravel() ** 2 + y.ravel() ** 2)
        boundary_hole = flat[radius > 0.025]
        hole = self.measure(boundary_hole)
        self.assertNotEqual(
            hole["geometric_predicates"]["OPEN_CAVITY"]["status"],
            "TRUE",
        )

    def test_adequate_solid_nonreceptacle_is_false(self):
        x, y = np.meshgrid(
            np.linspace(-0.045, 0.045, 24),
            np.linspace(-0.035, 0.035, 20),
        )
        top = np.column_stack(
            (x.ravel(), y.ravel(), np.full(x.size, 0.10))
        )
        rng = np.random.default_rng(20)
        body = rng.uniform(
            (-0.045, -0.035, 0.0),
            (0.045, 0.035, 0.09),
            (900, 3),
        )
        properties = self.measure(np.vstack((top, body)))
        predicate = properties["geometric_predicates"]["OPEN_CAVITY"]
        self.assertEqual(predicate["status"], "FALSE")

    def test_horizontal_thin_surface_is_planar_support(self):
        x, y = np.meshgrid(
            np.linspace(-0.10, 0.10, 30),
            np.linspace(-0.07, 0.07, 24),
        )
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        properties = self.measure(points)
        self.assertEqual(
            properties["geometric_predicates"]["PLANAR_SUPPORT"]["status"],
            "TRUE",
        )

    def test_property_records_contain_stage_local_provenance(self):
        properties = self.measure(cavity_cloud())
        records = [
            properties["centroid_world_m"],
            properties["dimensions_m"]["length"],
            properties["geometric_properties"]["opening_width_m"],
            properties["geometric_predicates"]["OPEN_CAVITY"],
        ]
        for record in records:
            self.assertEqual(record["source_stage"], 3)
            self.assertEqual(record["source_region"], "C2")
            self.assertTrue(record["measurement_cloud_path"].endswith("fused.ply"))
            self.assertNotIn("cumulative", record["measurement_cloud_path"])
            self.assertEqual(record["cloud_purpose"], "MEASUREMENT_EVIDENCE")
            self.assertTrue(record["contributing_camera_ids"])

    def test_pairwise_relations_use_cached_generic_geometry(self):
        tool_points = np.random.default_rng(9).uniform(
            (-0.10, -0.01, -0.005),
            (0.10, 0.01, 0.005),
            (500, 3),
        )
        tool = self.measure(tool_points)
        receptacle = self.measure(cavity_cloud())
        self.assertEqual(
            pairwise_relation_status(
                "INSERTABLE_IN", tool, receptacle, self.config
            ),
            "TRUE",
        )
        self.assertEqual(
            pairwise_relation_status(
                "REACHES_BOTTOM", tool, receptacle, self.config
            ),
            "TRUE",
        )
        insertable = pairwise_relation_evaluation(
            "INSERTABLE_IN", tool, receptacle, self.config
        )
        self.assertEqual(insertable["status"], "TRUE")
        self.assertIn("maximum_cross_section_m", insertable)
        self.assertIn("opening_width_m", insertable)
        self.assertIn("clearance_margin_m", insertable)
        self.assertEqual(
            insertable["inference_basis"], "GEOMETRY_ONLY"
        )

    def test_missing_target_geometry_makes_pairwise_unknown(self):
        tool = self.measure(
            np.linspace((0, 0, 0), (0.2, 0.01, 0.005), 100)
        )
        target = self.measure(
            np.zeros((2, 3)),
            quality_valid=False,
            cameras=(),
        )
        for relation in ("INSERTABLE_IN", "REACHES_BOTTOM"):
            self.assertEqual(
                pairwise_relation_status(
                    relation, tool, target, self.config
                ),
                "UNKNOWN",
            )


if __name__ == "__main__":
    unittest.main()
