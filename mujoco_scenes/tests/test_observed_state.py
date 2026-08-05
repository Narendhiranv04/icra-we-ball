import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mujoco_scenes.geometry_checker import (
    InspectionCameraCapture,
    MeasurementEvidence,
    ObjectPointCloud,
    PointCloudRun,
    RegionInspection,
)
from mujoco_scenes.geometry_properties import extract_object_properties
from mujoco_scenes.observed_state import (
    ObservedStateRun,
    _select_validated_semantic,
)


CAMERAS = (
    "inspection_left",
    "inspection_right",
    "inspection_top",
    "inspection_front",
    "inspection_close",
)


class FakeScene:
    scene_name = "synthetic_scene"
    has_robot = False

    def __init__(self):
        self.region_states = {
            region: {"open": False, "inspected": False}
            for region in ("C1", "C2", "D1")
        }
        self.instance_source_regions = {
            "kettle_instance": "countertop",
            "spoon_instance": "countertop",
            "fork_instance": "D1",
            "plate_instance": "C2",
            "bowl_instance": "C2",
            "mug_instance": "C1",
        }

    def get_region_observation_states(self):
        return {
            region: {
                "region_id": region,
                "open": state["open"],
                "inspected": state["inspected"],
            }
            for region, state in self.region_states.items()
        }

    def get_instance_source_region(self, instance_id):
        return self.instance_source_regions[instance_id]


def synthetic_points(instance_name, offset=(0, 0, 0)):
    rng = np.random.default_rng(sum(map(ord, instance_name)))
    points = rng.uniform(
        low=(-0.08, -0.02, -0.01),
        high=(0.08, 0.02, 0.01),
        size=(160, 3),
    ).astype(np.float32)
    return points + np.asarray(offset, dtype=np.float32)


def cavity_points():
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    rim = np.column_stack(
        (
            0.05 * np.cos(angles),
            0.04 * np.sin(angles),
            np.full(len(angles), 0.10),
        )
    )
    heights = np.linspace(0.02, 0.09, 5)
    aa, zz = np.meshgrid(angles, heights)
    wall = np.column_stack(
        (
            0.05 * np.cos(aa.ravel()),
            0.04 * np.sin(aa.ravel()),
            zz.ravel(),
        )
    )
    x, y = np.meshgrid(
        np.linspace(-0.018, 0.018, 12),
        np.linspace(-0.012, 0.012, 10),
    )
    interior = np.column_stack(
        (x.ravel(), y.ravel(), np.full(x.size, 0.015))
    )
    return np.vstack((rim, wall, interior)).astype(np.float32)


def make_evidence(
    instance_name,
    points,
    region,
    *,
    valid=True,
    object_kind="ignored_label",
):
    del object_kind
    points = np.asarray(points, dtype=np.float32)
    colors = np.tile(np.array([[80, 140, 210]], dtype=np.uint8), (len(points), 1))
    cameras = CAMERAS if valid else CAMERAS[:1]
    return MeasurementEvidence(
        instance_name=instance_name,
        measurement_points=points,
        measurement_colors=colors,
        contributing_camera_ids=tuple(cameras),
        points_by_camera={camera: points.copy() for camera in cameras},
        source_stage=None,
        source_region=region,
        measurement_cloud_path=None,
        measurement_quality={
            "quality_is_valid": valid,
            "status": "VALID" if valid else "INVALID",
            "reasons": [] if valid else ["INSUFFICIENT_OBJECT_CAMERA_COVERAGE"],
            "point_count": len(points),
            "raw_inside_point_count": len(points),
            "inside_fraction": 1.0,
            "contributing_camera_count": len(cameras),
            "outlier_points_removed": 0,
        },
    )


def fake_capture(camera_id):
    return InspectionCameraCapture(
        camera_id=camera_id,
        model_camera_name=f"model_{camera_id}",
        position_world_m=np.array([0.0, -1.0, 1.0]),
        rotation_world_from_camera=np.eye(3),
        target_world_m=np.array([0.0, 0.0, 0.5]),
        intrinsics=np.eye(3),
        fovy_degrees=60.0,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2), dtype=np.float32),
        segmentation=np.zeros((2, 2, 2), dtype=np.int32),
        validation={"usable": True, "reasons": []},
    )


def inspection_run(region, accepted, rejected=()):
    evidence_clouds = {
        item.instance_name: item for item in accepted
    }
    clouds = {
        item.instance_name: ObjectPointCloud(
            instance_name=item.instance_name,
            object_kind="ignored",
            points=item.measurement_points,
            colors=item.measurement_colors,
            pixels_by_camera={
                camera: len(item.measurement_points)
                for camera in item.contributing_camera_ids
            },
        )
        for item in accepted
    }
    rejected_clouds = {
        name: ObjectPointCloud(
            instance_name=name,
            object_kind="ignored",
            points=points,
            colors=np.zeros((len(points), 3), dtype=np.uint8),
            pixels_by_camera={camera: 0 for camera in CAMERAS},
        )
        for name, points in rejected
    }
    cameras = {camera: fake_capture(camera) for camera in CAMERAS}
    inspection = RegionInspection(
        region_id=region,
        rig_config={},
        cameras=cameras,
        evidence_clouds=evidence_clouds,
        rejected_clouds=rejected_clouds,
        metadata={
            "region_id": region,
            "region_open": True,
            "settle_steps": 10,
            "inspection_volume": {
                "minimum_world_m": [-1.0, -1.0, -1.0],
                "maximum_world_m": [1.0, 1.0, 1.0],
                "boundary_margin_m": 0.01,
            },
        },
        quality={
            "valid_camera_count": 5,
            "capture_quality_is_valid": True,
            "_rejected_instance_reasons": {
                name: ["SOURCE_REGION_MISMATCH"]
                for name, _points in rejected
            },
        },
    )
    return PointCloudRun(
        clouds=clouds,
        cameras=CAMERAS,
        width=64,
        height=48,
        timings_seconds={"total": 0.01},
        inspection=inspection,
    )


class ObservedStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run"
        self.scene = FakeScene()
        self.session = ObservedStateRun(
            self.run_dir,
            scene_name=self.scene.scene_name,
            region_ids=self.scene.region_states,
        )
        self.kettle = make_evidence(
            "kettle_instance",
            synthetic_points("kettle_instance"),
            "INITIAL",
        )
        self.spoon = make_evidence(
            "spoon_instance",
            synthetic_points("spoon_instance", (0.3, 0, 0)),
            "INITIAL",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _initial(self):
        return self.session.update_from_inspection_run(
            self.scene,
            inspection_run("INITIAL", (self.kettle, self.spoon)),
            stage_label="initial",
        )

    def test_initial_objects_use_initial_stage_evidence_and_hidden_is_absent(self):
        self._initial()
        objects = self.session.registry["objects"]
        self.assertEqual(set(objects), {"object_0001", "object_0002"})
        self.assertNotIn("mug_instance", json.dumps(self.session.registry))
        for record in objects.values():
            self.assertEqual(record["last_property_update_stage"], 0)
            self.assertEqual(record["last_property_source_region"], "INITIAL")
            self.assertIn("/evidence/", record["measurement_cloud_path"])
            self.assertTrue(record["measurement_cloud_path"].endswith("fused.ply"))
            self.assertNotIn("cumulative", record["measurement_cloud_path"])

    def test_c2_updates_only_region_evidence_and_preserves_tabletop_cache(self):
        self._initial()
        tabletop_before = deepcopy_json(
            self.session.registry["objects"]["object_0001"]
        )
        self.scene.region_states["C2"] = {"open": True, "inspected": True}
        bowl = make_evidence("bowl_instance", cavity_points(), "C2")
        self.session.update_from_inspection_run(
            self.scene,
            inspection_run(
                "C2",
                (bowl,),
                rejected=(
                    (
                        "kettle_instance",
                        synthetic_points("kettle_instance"),
                    ),
                ),
            ),
            stage_label="after_C2",
            region_opened="C2",
        )
        objects = self.session.registry["objects"]
        self.assertEqual(len(objects), 3)
        tabletop_after = objects["object_0001"]
        self.assertEqual(
            tabletop_after["last_property_update_stage"],
            tabletop_before["last_property_update_stage"],
        )
        self.assertEqual(
            tabletop_after["observation_count"],
            tabletop_before["observation_count"],
        )
        self.assertEqual(
            tabletop_after["measurement_cloud_path"],
            tabletop_before["measurement_cloud_path"],
        )
        new_record = objects["object_0003"]
        self.assertEqual(new_record["source_region"], "C2")
        self.assertEqual(new_record["first_seen_stage"], 1)

    def test_reinspection_updates_same_instance_without_duplicate(self):
        self._initial()
        self.scene.region_states["D1"] = {"open": True, "inspected": True}
        fork = make_evidence(
            "fork_instance", synthetic_points("fork_instance"), "D1"
        )
        self.session.update_from_inspection_run(
            self.scene,
            inspection_run("D1", (fork,)),
            stage_label="after_D1",
            region_opened="D1",
        )
        self.session.update_from_inspection_run(
            self.scene,
            inspection_run("D1", (fork,)),
            stage_label="reinspect_D1",
            region_opened="D1",
        )
        self.assertEqual(len(self.session.registry["objects"]), 3)
        record = self.session.registry["objects"]["object_0003"]
        self.assertEqual(record["observation_count"], 2)
        self.assertEqual(record["first_seen_stage"], 1)
        self.assertEqual(record["last_property_update_stage"], 2)

    def test_invalid_reinspection_keeps_previous_cached_properties(self):
        self._initial()
        before = deepcopy_json(self.session.registry["objects"]["object_0001"])
        invalid = make_evidence(
            "kettle_instance",
            synthetic_points("kettle_instance", (0.1, 0, 0)),
            "INITIAL",
            valid=False,
        )
        self.session.update_from_inspection_run(
            self.scene,
            inspection_run("INITIAL", (invalid,)),
            stage_label="invalid_reinspection",
        )
        after = self.session.registry["objects"]["object_0001"]
        self.assertEqual(
            after["last_property_update_stage"],
            before["last_property_update_stage"],
        )
        self.assertEqual(
            after["measurement_cloud_path"],
            before["measurement_cloud_path"],
        )
        self.assertEqual(
            after["geometric_properties"],
            before["geometric_properties"],
        )

    def test_extractor_receives_only_fused_measurement_evidence(self):
        with patch(
            "mujoco_scenes.observed_state.extract_object_properties",
            wraps=extract_object_properties,
        ) as spy:
            self._initial()
        self.assertGreater(spy.call_count, 0)
        for call in spy.call_args_list:
            supplied = call.args[0]
            self.assertIsInstance(supplied, MeasurementEvidence)
            self.assertEqual(
                supplied.cloud_purpose, "MEASUREMENT_EVIDENCE"
            )
            self.assertTrue(supplied.measurement_cloud_path.endswith("fused.ply"))
            self.assertNotIn("cumulative", supplied.measurement_cloud_path)
            self.assertNotIn("combined_cloud", supplied.measurement_cloud_path)

    def test_scene_wide_point_cloud_run_is_rejected(self):
        plain = PointCloudRun(
            clouds={},
            cameras=(),
            width=64,
            height=48,
            timings_seconds={},
        )
        with self.assertRaises(TypeError):
            self.session.update_from_point_cloud_run(
                self.scene, plain, stage_label="initial"
            )

    def test_snapshot_contains_inspection_debug_and_visualization_outputs(self):
        stage = self._initial()
        required = {
            "combined_cloud.ply",
            "region_combined_cloud.ply",
            "inspection_metadata.json",
            "inspection_quality.json",
            "properties.json",
            "graph.json",
            "witness.json",
            "all_observed_pair_relations.json",
            "pointcloud.png",
            "graph.png",
            "overview.png",
        }
        self.assertTrue(required.issubset({path.name for path in stage.iterdir()}))
        self.assertTrue((self.run_dir / "graph_growth.gif").exists())
        for object_id in ("object_0001", "object_0002"):
            self.assertTrue(
                (stage / "evidence" / object_id / "fused.ply").exists()
            )
            self.assertTrue(
                (
                    self.run_dir
                    / "objects"
                    / object_id
                    / "cumulative_visualization.ply"
                ).exists()
            )

    def test_all_observed_objects_are_paired_before_role_binding(self):
        stage = self._initial()
        payload = json.loads(
            (stage / "all_observed_pair_relations.json").read_text()
        )
        self.assertEqual(
            payload["pairing_scope"],
            "ALL_OBSERVED_ORDERED_OBJECT_PAIRS",
        )
        self.assertEqual(
            payload["role_binding_phase"],
            "AFTER_PAIRWISE_GEOMETRY",
        )
        self.assertEqual(payload["observed_object_ids"], [
            "object_0001", "object_0002"
        ])
        self.assertEqual(
            payload["ordered_distinct_object_pair_count"], 2
        )
        self.assertEqual(payload["relation_evaluation_count"], 4)
        self.assertEqual(
            {
                (
                    item["source_object_id"],
                    item["target_object_id"],
                )
                for item in payload["relations"]
            },
            {
                ("object_0001", "object_0002"),
                ("object_0002", "object_0001"),
            },
        )

    def test_semantic_role_scoping_skips_irrelevant_binary_geometry(self):
        self._initial()
        task = (
            Path(__file__).parents[1]
            / "configs"
            / "ablation3_multi_target.yaml"
        )
        session = ObservedStateRun(
            Path(self.temporary.name) / "semantic_scoped",
            scene_name=self.scene.scene_name,
            region_ids=self.scene.region_states,
            task_requirements=task,
            pairing_strategy="semantic_role_scoped",
        )
        template = deepcopy_json(
            self.session.registry["objects"]["object_0001"]
        )
        session.registry["objects"] = {}
        for index, label in enumerate(("spoon", "cup", "marker"), 1):
            record = deepcopy_json(template)
            object_id = f"object_{index:04d}"
            record["object_id"] = object_id
            record["semantics"] = {
                "validated": {
                    "status": "SUPPORTED",
                    "canonical_label": label,
                    "mean_confidence": 0.9,
                }
            }
            session.registry["objects"][object_id] = record
        result = {"status": "TRUE", "pass_margin_m": 0.01}
        with patch(
            "mujoco_scenes.observed_state.pairwise_relation_evaluation",
            return_value=result,
        ) as evaluator:
            graph = session._build_graph(
                self.scene.get_region_observation_states(), {}
            )
        # Only spoon -> cup is semantically compatible; it is evaluated for
        # INSERTABLE_IN and REACHES_BOTTOM. Marker and reverse pairs are pruned.
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(
            graph["pairing"]["possible_ordered_pair_count"], 6
        )
        self.assertEqual(
            graph["pairing"]["relation_evaluation_count"], 2
        )
        self.assertEqual(
            graph["pairing"]["skipped_relation_pair_count"], 10
        )

    def test_valid_initial_object_can_complete_global_task(self):
        task = Path(__file__).parents[1] / "configs" / "s1_find_open_receptacle.yaml"
        session = ObservedStateRun(
            Path(self.temporary.name) / "global",
            scene_name=self.scene.scene_name,
            region_ids=self.scene.region_states,
            task_requirements=task,
        )
        initial_receptacle = make_evidence(
            "kettle_instance", cavity_points(), "INITIAL"
        )
        session.update_from_inspection_run(
            self.scene,
            inspection_run("INITIAL", (initial_receptacle,)),
            stage_label="initial",
        )
        self.assertEqual(session.latest_witness["status"], "COMPLETE")
        self.assertEqual(
            session.latest_witness["selected_witness"],
            {"open_receptacle": ["object_0001"]},
        )

    def test_weak_reobservation_does_not_replace_validated_semantics(self):
        strong = {
            "status": "SUPPORTED",
            "canonical_label": "fork",
            "source_stage": 2,
            "quality": {
                "supporting_view_count": 4,
                "mean_confidence": 0.75,
                "winning_label_margin": 2.0,
            },
        }
        weak = {
            "status": "UNKNOWN",
            "canonical_label": None,
            "source_stage": 3,
            "quality": {
                "supporting_view_count": 1,
                "mean_confidence": 0.20,
                "winning_label_margin": 0.0,
            },
        }
        selected, replaced = _select_validated_semantic(strong, weak)
        self.assertFalse(replaced)
        self.assertEqual(selected, strong)
        self.assertIsNot(selected, strong)


def deepcopy_json(value):
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
