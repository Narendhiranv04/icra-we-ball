import tempfile
import unittest
from pathlib import Path

from mujoco_scenes.geometry_checker import GeometryChecker
from mujoco_scenes.scene_loader import CONTAINER_JOINTS, KitchenScene


class ObservedStateSceneSmokeTests(unittest.TestCase):
    def test_existing_open_all_and_point_cloud_export_remain_functional(self):
        scene = KitchenScene("S1_coffee_missing_mug", include_robot=False)
        scene.set_all_containers_open_snapshot()
        self.assertEqual(
            scene.state.opened_containers,
            set(CONTAINER_JOINTS),
        )
        self.assertTrue(
            all(scene.state.container_open_state.values())
        )
        with tempfile.TemporaryDirectory() as directory:
            run = GeometryChecker(scene, width=160, height=120).run(directory)
            output = Path(directory)
            self.assertTrue((output / "all_visible_objects.ply").exists())
            self.assertTrue((output / "manifest.json").exists())
        self.assertIn("mug", {cloud.object_kind for cloud in run.clouds.values()})
        self.assertGreater(run.total_points, 0)


if __name__ == "__main__":
    unittest.main()
