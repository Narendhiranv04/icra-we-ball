import unittest

import mujoco

from mujoco_scenes.geometry_checker import GeometryChecker, load_inspection_rig_config
from mujoco_scenes.workshop_alternatives import evaluate_ranked_alternatives
from mujoco_scenes.workshop_scene import (
    WORKSHOP_CAMERAS,
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
        self.assertNotIn("workshop_flat_driver", initial_names)
        self.assertIsNone(
            scene.get_instance_source_region("workshop_flat_driver")
        )
        revealed = scene.open_container("LEFT_DRAWER")
        self.assertIn("workshop_flat_driver", revealed)
        self.assertEqual(
            scene.get_instance_source_region("workshop_flat_driver"),
            "LEFT_DRAWER",
        )

    def test_workshop_has_five_views_for_each_observation_stage(self):
        config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
        self.assertEqual(
            config["inspection_sequence"], ["LEFT_DRAWER", "RIGHT_DRAWER"]
        )
        self.assertEqual(set(config["camera_slots"].values()), set(WORKSHOP_CAMERAS))
        for region in config["regions"].values():
            self.assertEqual(len(region["cameras"]), 5)

    def test_open_drawer_produces_fresh_region_gated_rgbd_evidence(self):
        scene = WorkshopScene("none")
        scene.open_container("RIGHT_DRAWER")
        run = GeometryChecker(scene, width=320, height=240).run_region_inspection(
            "RIGHT_DRAWER"
        )
        self.assertGreater(len(run.clouds["workshop_phillips_driver"].points), 20)
        self.assertGreater(len(run.clouds["workshop_medium_screw"].points), 20)


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
