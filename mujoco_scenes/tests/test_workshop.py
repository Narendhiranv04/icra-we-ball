import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco

from mujoco_scenes.geometry_checker import GeometryChecker, load_inspection_rig_config
from mujoco_scenes.workshop_alternatives import evaluate_ranked_alternatives
from mujoco_scenes.workshop_pointcloud import run_workshop_pointcloud
from mujoco_scenes.workshop_scene import (
    WORKSHOP_CAMERAS,
    WORKSHOP_FUNCTIONAL_REGIONS,
    WORKSHOP_INSPECTION_RIG_CONFIG,
    WorkshopScene,
)


TARGET = {
    "hole_diameter_m": 0.007,
    "joint_depth_m": 0.030,
    "radial_clearance_m": 0.0005,
}
OBJECTS = [
    {
        "object_id": "hammer",
        "functions": ["can_hammer"],
        "geometry": {"face_width_m": 0.040},
        "source_region": "workbench",
    },
    {
        "object_id": "large_nail",
        "functions": ["can_fasten"],
        "geometry": {
            "diameter_m": 0.010,
            "length_m": 0.090,
            "head_width_m": 0.022,
        },
        "source_region": "workbench",
    },
    {
        "object_id": "flat_driver",
        "functions": ["can_screw"],
        "geometry": {"tip_profile": "flat", "tip_width_m": 0.004},
        "source_region": "LEFT_DRAWER",
    },
    {
        "object_id": "short_screw",
        "functions": ["can_fasten"],
        "geometry": {
            "diameter_m": 0.005,
            "length_m": 0.015,
            "recess_profile": "flat",
            "recess_width_m": 0.005,
        },
        "source_region": "LEFT_DRAWER",
    },
    {
        "object_id": "phillips_driver",
        "functions": ["can_screw"],
        "geometry": {
            "tip_profile": "phillips_2",
            "tip_width_m": 0.004,
        },
        "source_region": "RIGHT_DRAWER",
    },
    {
        "object_id": "medium_screw",
        "functions": ["can_fasten"],
        "geometry": {
            "diameter_m": 0.005,
            "length_m": 0.050,
            "recess_profile": "phillips_2",
            "recess_width_m": 0.0045,
        },
        "source_region": "RIGHT_DRAWER",
    },
]


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
        self.assertIn("workshop_cabinet_key", revealed)
        self.assertNotIn("workshop_power_driver", revealed)
        self.assertEqual(
            scene.get_instance_source_region("workshop_manual_driver"),
            "LEFT_DRAWER",
        )
        self.assertIsNone(
            scene.get_instance_source_region("workshop_power_driver")
        )

    def test_scene_exposes_fixture_tray_and_locked_cabinet(self):
        scene = WorkshopScene("none")
        self.assertEqual(
            WORKSHOP_FUNCTIONAL_REGIONS,
            (
                "FRAME_FIXTURE",
                "SCREW_STAGING_TRAY",
                "LOCKED_TOOL_CABINET",
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
            "locked_cabinet_door_hinge",
        )
        self.assertGreaterEqual(cabinet_joint_id, 0)
        self.assertEqual(
            scene.model.jnt_type[cabinet_joint_id],
            mujoco.mjtJoint.mjJNT_HINGE,
        )
        task_state = scene.get_task_scene_state()
        self.assertFalse(task_state["joint_repaired"])
        self.assertTrue(task_state["locked_cabinet"]["locked"])
        self.assertFalse(task_state["locked_cabinet"]["open"])
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

    def test_locked_cabinet_requires_observed_matching_key(self):
        scene = WorkshopScene("none")
        with self.assertRaisesRegex(RuntimeError, "locked"):
            scene.open_container("LOCKED_CABINET")
        with self.assertRaisesRegex(ValueError, "must be observed"):
            scene.unlock_container("LOCKED_CABINET", "workshop_cabinet_key")

        scene.open_container("LEFT_DRAWER")
        scene.unlock_container("LOCKED_CABINET", "workshop_cabinet_key")
        revealed = scene.open_container("LOCKED_CABINET")
        self.assertIn("workshop_power_driver", revealed)
        self.assertIn("workshop_long_screw", revealed)
        task_state = scene.get_task_scene_state()
        self.assertFalse(task_state["locked_cabinet"]["locked"])
        self.assertTrue(task_state["locked_cabinet"]["open"])
        door_joint_id = mujoco.mj_name2id(
            scene.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "locked_cabinet_door_hinge",
        )
        door_qpos_address = scene.model.jnt_qposadr[door_joint_id]
        self.assertGreater(scene.data.qpos[door_qpos_address], 1.0)
        lock_geom_id = mujoco.mj_name2id(
            scene.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "locked_cabinet_lock",
        )
        self.assertGreater(
            scene.model.geom_rgba[lock_geom_id, 1],
            scene.model.geom_rgba[lock_geom_id, 0],
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
            config["inspection_sequence"], ["LEFT_DRAWER", "LOCKED_CABINET"]
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
            "workshop_cabinet_key",
        ):
            self.assertGreater(len(left_run.clouds[instance_name].points), 20)

        scene.unlock_container("LOCKED_CABINET", "workshop_cabinet_key")
        scene.open_container("LOCKED_CABINET")
        right_run = checker.run_region_inspection("LOCKED_CABINET")
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
                ["INITIAL", "LEFT_DRAWER", "LOCKED_CABINET"],
            )
            self.assertEqual(manifest["segmentation"], "oracle")
            self.assertTrue((output / "manifest.json").is_file())
            drawer_objects = {
                item["debug_instance_id"]
                for item in manifest["stages"][1]["objects"]
            }
            self.assertIn("workshop_cabinet_key", drawer_objects)
            for stage in manifest["stages"]:
                stage_dir = output / stage["directory"]
                self.assertTrue((stage_dir / "stage_summary.json").is_file())
                self.assertTrue((stage_dir / stage["combined_ply"]).is_file())
                self.assertGreater(stage["total_point_count"], 20)
            self.assertFalse(
                scene.get_task_scene_state()["locked_cabinet"]["locked"]
            )


class WorkshopAlternativeTests(unittest.TestCase):
    def test_joint_reasoning_rejects_two_near_misses_then_selects_screw(self):
        proposals = [
            {
                "rank": 1,
                "method": "nail",
                "tool_object_id": "hammer",
                "fastener_object_id": "large_nail",
            },
            {
                "rank": 2,
                "method": "screw",
                "tool_object_id": "flat_driver",
                "fastener_object_id": "short_screw",
            },
            {
                "rank": 3,
                "method": "screw",
                "tool_object_id": "phillips_driver",
                "fastener_object_id": "medium_screw",
            },
        ]
        result = evaluate_ranked_alternatives(
            observed_objects=OBJECTS,
            target_geometry=TARGET,
            ranked_proposals=proposals,
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["selected"]["rank"], 3)
        self.assertFalse(result["evaluated"][0]["geometry_checks"]["fits_hole"])
        self.assertFalse(
            result["evaluated"][1]["geometry_checks"]["reaches_joint"]
        )

    def test_unobserved_object_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside visible state"):
            evaluate_ranked_alternatives(
                observed_objects=OBJECTS[:2],
                target_geometry=TARGET,
                ranked_proposals=[
                    {
                        "rank": 1,
                        "method": "screw",
                        "tool_object_id": "invented_driver",
                        "fastener_object_id": "large_nail",
                    }
                ],
            )

    def test_accepts_at_most_three_fm_alternatives(self):
        proposal = {
            "method": "nail",
            "tool_object_id": "hammer",
            "fastener_object_id": "large_nail",
        }
        with self.assertRaisesRegex(ValueError, "maximum is 3"):
            evaluate_ranked_alternatives(
                observed_objects=OBJECTS,
                target_geometry=TARGET,
                ranked_proposals=[
                    {**proposal, "rank": rank} for rank in range(1, 5)
                ],
            )


if __name__ == "__main__":
    unittest.main()
