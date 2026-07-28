import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mujoco_scenes.geometry_checker import ObjectPointCloud, PointCloudRun
from mujoco_scenes.observed_state import ObservedStateRun


class FakeScene:
    scene_name = "synthetic_scene"

    def __init__(self):
        self.region_states = {
            region: {"open": False, "inspected": False}
            for region in ("C1", "D1")
        }
        self.instance_source_regions = {
            "kettle_instance": "countertop",
            "spoon_instance": "countertop",
            "fork_instance": "D1",
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


def synthetic_cloud(instance_name, category, offset=(0, 0, 0)):
    rng = np.random.default_rng(sum(map(ord, instance_name)))
    points = rng.uniform(
        low=(-0.08, -0.02, -0.01),
        high=(0.08, 0.02, 0.01),
        size=(120, 3),
    ).astype(np.float32)
    points += np.asarray(offset, dtype=np.float32)
    colors = np.tile(np.array([[80, 140, 210]], dtype=np.uint8), (len(points), 1))
    return ObjectPointCloud(
        instance_name=instance_name,
        object_kind=category,
        points=points,
        colors=colors,
        pixels_by_camera={"cam1": 60, "cam2": 60},
    )


def point_cloud_run(*clouds):
    return PointCloudRun(
        clouds={cloud.instance_name: cloud for cloud in clouds},
        cameras=("cam1", "cam2"),
        width=64,
        height=48,
        timings_seconds={"total": 0.01},
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
        self.kettle = synthetic_cloud("kettle_instance", "kettle")
        self.spoon = synthetic_cloud("spoon_instance", "spoon", (0.3, 0, 0))

    def tearDown(self):
        self.temporary.cleanup()

    def _initial(self):
        return self.session.update_from_point_cloud_run(
            self.scene,
            point_cloud_run(self.kettle, self.spoon),
            stage_label="initial",
        )

    def test_initial_visible_objects_are_added_but_hidden_object_is_absent(self):
        self._initial()
        objects = self.session.registry["objects"]
        self.assertEqual(
            {record["category"] for record in objects.values()},
            {"kettle", "spoon"},
        )
        self.assertNotIn(
            "mug_instance", self.session.registry["instance_index"]
        )
        graph = json.loads((self.run_dir / "observed_graph.json").read_text())
        graph_instances = {
            node["attributes"].get("instance_id")
            for node in graph["nodes"]
            if node["type"] == "object"
        }
        self.assertNotIn("mug_instance", graph_instances)

    def test_opening_one_region_adds_only_new_visible_objects_and_keeps_old(self):
        self._initial()
        self.scene.region_states["D1"] = {"open": True, "inspected": True}
        fork = synthetic_cloud("fork_instance", "fork", (-0.3, 0, 0))
        self.session.update_from_point_cloud_run(
            self.scene,
            point_cloud_run(self.kettle, self.spoon, fork),
            stage_label="after_D1",
            region_opened="D1",
        )
        objects = self.session.registry["objects"]
        self.assertEqual(
            {record["category"] for record in objects.values()},
            {"kettle", "spoon", "fork"},
        )
        self.assertNotIn("mug_instance", self.session.registry["instance_index"])
        self.assertIn("spoon_instance", self.session.registry["instance_index"])
        graph = json.loads((self.run_dir / "observed_graph.json").read_text())
        object_nodes = [node for node in graph["nodes"] if node["type"] == "object"]
        self.assertEqual(len(object_nodes), 3)

    def test_reobserving_instance_updates_without_duplicate(self):
        self._initial()
        spoon_id = self.session.registry["instance_index"]["spoon_instance"]
        self.scene.region_states["D1"] = {"open": True, "inspected": True}
        self.session.update_from_point_cloud_run(
            self.scene,
            point_cloud_run(self.kettle, self.spoon),
            stage_label="manual",
        )
        self.assertEqual(
            self.session.registry["instance_index"]["spoon_instance"], spoon_id
        )
        spoon_records = [
            record
            for record in self.session.registry["objects"].values()
            if record["instance_id"] == "spoon_instance"
        ]
        self.assertEqual(len(spoon_records), 1)
        self.assertEqual(spoon_records[0]["observation_count"], 2)
        self.assertEqual(spoon_records[0]["first_seen_stage"], 0)
        self.assertEqual(spoon_records[0]["last_seen_stage"], 1)

    def test_snapshot_is_saved_for_initial_and_every_opening(self):
        initial_dir = self._initial()
        self.scene.region_states["D1"] = {"open": True, "inspected": True}
        fork = synthetic_cloud("fork_instance", "fork")
        opened_dir = self.session.update_from_point_cloud_run(
            self.scene,
            point_cloud_run(self.kettle, self.spoon, fork),
            stage_label="after_D1",
            region_opened="D1",
        )
        required = {
            "combined_cloud.ply",
            "properties.json",
            "graph.json",
            "pointcloud.png",
            "graph.png",
            "overview.png",
        }
        self.assertTrue(required.issubset({path.name for path in initial_dir.iterdir()}))
        self.assertTrue(required.issubset({path.name for path in opened_dir.iterdir()}))
        self.assertTrue((self.run_dir / "graph_growth.gif").exists())


if __name__ == "__main__":
    unittest.main()
