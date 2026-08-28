import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco

from mujoco_scenes.geometry_checker import GeometryChecker, load_inspection_rig_config
from mujoco_scenes.workshop_pointcloud import run_workshop_pointcloud
from mujoco_scenes.workshop_scene import (
    WORKSHOP_CAMERAS,
    WORKSHOP_FUNCTIONAL_REGIONS,
    WORKSHOP_INSPECTION_RIG_CONFIG,
    WorkshopScene,
)


class WorkshopSceneTests(unittest.TestCase):
    def test_scene_compiles_with_and_without_google_robot(self):
        for robot in ("none", "google"):
            scene = WorkshopScene(robot)
            self.assertEqual(scene.has_robot, robot == "google")
            for camera_name in WORKSHOP_CAMERAS:
                self.assertGreaterEqual(
                    mujoco.mj_name2id(
                        scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                    ),
                    0,
                )

    def test_drawer_objects_are_revealed_only_after_inspection(self):
        scene = WorkshopScene("none")
        initial_names = dict(scene.get_visible_object_instances())
        self.assertNotIn("workshop_manual_driver", initial_names)
        self.assertNotIn("workshop_power_driver", initial_names)
        self.assertIsNone(
            scene.get_instance_source_region("workshop_manual_driver")
        )
        revealed = scene.open_container("LEFT_DRAWER")
        self.assertIn("workshop_manual_driver", revealed)
        self.assertIn("workshop_short_screw", revealed)
        self.assertNotIn("workshop_power_driver", revealed)
        self.assertEqual(
            scene.get_instance_source_region("workshop_manual_driver"),
            "LEFT_DRAWER",
        )
        self.assertIsNone(
            scene.get_instance_source_region("workshop_power_driver")
        )

    def test_scene_exposes_fixture_tray_and_closed_tool_cabinet(self):
        scene = WorkshopScene("none")
        self.assertEqual(
            WORKSHOP_FUNCTIONAL_REGIONS,
            (
                "FRAME_FIXTURE",
                "SCREW_STAGING_TRAY",
                "TOOL_CABINET",
            ),
        )

        initial_names = dict(scene.get_visible_object_instances())
        self.assertEqual(
            initial_names["workshop_frame_joint"],
            "fixture_held_frame_joint",
        )
        self.assertEqual(
            initial_names["workshop_joint_seal"],
            "protective_joint_seal",
        )
        cabinet_joint_id = mujoco.mj_name2id(
            scene.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "tool_cabinet_door_hinge",
        )
        self.assertGreaterEqual(cabinet_joint_id, 0)
        self.assertEqual(
            scene.model.jnt_type[cabinet_joint_id],
            mujoco.mjtJoint.mjJNT_HINGE,
        )
        task_state = scene.get_task_scene_state()
        self.assertFalse(task_state["joint_repaired"])
        self.assertFalse(task_state["tool_cabinet"]["open"])
        self.assertFalse(
            scene.get_region_observation_states()["LEFT_DRAWER"]["open"]
        )
        self.assertFalse(task_state["joint_access"]["clear"])
        self.assertEqual(
            task_state["joint_access"]["covered_by"],
            "workshop_joint_seal",
        )

    def test_joint_seal_is_stable_and_can_be_staged_in_tray(self):
        scene = WorkshopScene("none")
        seal_joint_id = mujoco.mj_name2id(
            scene.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "workshop_joint_seal_free",
        )
        self.assertGreaterEqual(seal_joint_id, 0)
        self.assertEqual(
            scene.model.jnt_type[seal_joint_id],
            mujoco.mjtJoint.mjJNT_FREE,
        )
        scene.move_joint_seal_to_tray()
        state = scene.get_task_scene_state()
        self.assertTrue(state["joint_access"]["clear"])
        self.assertEqual(state["joint_seal_location"], "SCREW_STAGING_TRAY")
        qpos_address = scene.model.jnt_qposadr[seal_joint_id]
        self.assertAlmostEqual(scene.data.qpos[qpos_address], -0.84, places=2)
        self.assertAlmostEqual(scene.data.qpos[qpos_address + 1], 0.19, places=2)

    def test_tool_cabinet_opens_without_a_lock_and_closes_again(self):
        scene = WorkshopScene("none")
        revealed = scene.open_container("TOOL_CABINET")
        self.assertIn("workshop_power_driver", revealed)
        self.assertIn("workshop_long_screw", revealed)
        task_state = scene.get_task_scene_state()
        self.assertTrue(task_state["tool_cabinet"]["open"])
        door_joint_id = mujoco.mj_name2id(
            scene.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "tool_cabinet_door_hinge",
        )
        door_qpos_address = scene.model.jnt_qposadr[door_joint_id]
        self.assertGreater(scene.data.qpos[door_qpos_address], 1.0)
        self.assertEqual(
            mujoco.mj_name2id(
                scene.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "locked_cabinet_lock",
            ),
            -1,
        )

        scene.close_container("TOOL_CABINET")
        self.assertFalse(scene.get_task_scene_state()["tool_cabinet"]["open"])
        self.assertAlmostEqual(scene.data.qpos[door_qpos_address], 0.0, places=2)
        self.assertEqual(
            mujoco.mj_name2id(
                scene.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "workshop_cabinet_key",
            ),
            -1,
        )

    def test_short_and_long_screws_are_geometrically_distinct(self):
        scene = WorkshopScene("none")
        short_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_GEOM, "short_screw_shank"
        )
        long_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_GEOM, "long_screw_shank"
        )
        self.assertGreaterEqual(short_id, 0)
        self.assertGreaterEqual(long_id, 0)
        self.assertLess(
            scene.model.geom_size[short_id, 1],
            scene.model.geom_size[long_id, 1],
        )

    def test_workshop_has_five_views_for_each_observation_stage(self):
        config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
        self.assertEqual(
            config["inspection_sequence"], ["LEFT_DRAWER", "TOOL_CABINET"]
        )
        self.assertEqual(set(config["camera_slots"].values()), set(WORKSHOP_CAMERAS))
        for region in config["regions"].values():
            self.assertEqual(len(region["cameras"]), 5)

    def test_open_drawers_produce_fresh_region_gated_rgbd_evidence(self):
        scene = WorkshopScene("none")
        checker = GeometryChecker(scene, width=320, height=240)
        initial_run = checker.run_region_inspection("INITIAL")
        self.assertGreater(
            len(initial_run.clouds["workshop_joint_seal"].points), 20
        )
        scene.open_container("LEFT_DRAWER")
        left_run = checker.run_region_inspection("LEFT_DRAWER")
        for instance_name in (
            "workshop_manual_driver",
            "workshop_short_screw",
        ):
            self.assertGreater(len(left_run.clouds[instance_name].points), 20)

        scene.close_container("LEFT_DRAWER")
        scene.open_container("TOOL_CABINET")
        right_run = checker.run_region_inspection("TOOL_CABINET")
        for instance_name in (
            "workshop_power_driver",
            "workshop_long_screw",
        ):
            self.assertGreater(len(right_run.clouds[instance_name].points), 20)

    def test_workshop_pointcloud_runner_captures_all_regions(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "workshop_run"
            scene, manifest = run_workshop_pointcloud(
                output,
                robot="none",
                width=320,
                height=240,
            )

            self.assertEqual(
                [stage["region_id"] for stage in manifest["stages"]],
                ["INITIAL", "LEFT_DRAWER", "TOOL_CABINET"],
            )
            self.assertEqual(manifest["segmentation"], "oracle")
            self.assertTrue((output / "manifest.json").is_file())
            for stage in manifest["stages"]:
                stage_dir = output / stage["directory"]
                self.assertTrue((stage_dir / "stage_summary.json").is_file())
                self.assertTrue((stage_dir / stage["combined_ply"]).is_file())
                self.assertGreater(stage["total_point_count"], 20)
            states = scene.get_region_observation_states()
            self.assertFalse(states["LEFT_DRAWER"]["open"])
            self.assertFalse(states["TOOL_CABINET"]["open"])
            self.assertFalse(
                manifest["final_region_states"]["LEFT_DRAWER"]["open"]
            )
            self.assertFalse(
                manifest["final_region_states"]["TOOL_CABINET"]["open"]
            )
            self.assertFalse(scene.get_task_scene_state()["tool_cabinet"]["open"])

if __name__ == "__main__":
    unittest.main()
