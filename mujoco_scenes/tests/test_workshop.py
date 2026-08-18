"""Comprehensive test suite for the realistic Workshop (W1) benchmark."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from mujoco_scenes.geometry_checker import GeometryChecker, load_inspection_rig_config
from mujoco_scenes.workshop_alternatives import evaluate_ranked_alternatives
from mujoco_scenes.workshop_pointcloud import run_workshop_pointcloud
from mujoco_scenes.workshop_scene import (
    WORKSHOP_CAMERAS,
    WORKSHOP_FUNCTIONAL_PARTS_CONTAINERS,
    WORKSHOP_FUNCTIONAL_WORK_SURFACES,
    WORKSHOP_INSPECTION_RIG_CONFIG,
    WORKSHOP_REGIONS,
    WorkshopScene,
    _load_workshop_variants_config,
)


TARGET = {
    "hole_diameter_m": 0.007,
    "joint_depth_m": 0.030,
    "radial_clearance_m": 0.0005,
}

SAMPLE_OBJECTS = [
    {
        "object_id": "hammer",
        "functions": ["can_hammer"],
        "geometry": {"face_width_m": 0.040, "bounding_area_m2": 0.03},
        "source_region": "workbench",
    },
    {
        "object_id": "large_nail",
        "functions": ["can_fasten"],
        "geometry": {
            "diameter_m": 0.010,
            "length_m": 0.090,
            "head_width_m": 0.022,
            "bounding_area_m2": 0.005,
        },
        "source_region": "workbench",
    },
    {
        "object_id": "flat_driver",
        "functions": ["can_drive_screw"],
        "geometry": {
            "tip_profile": "SLOTTED",
            "tip_width_m": 0.004,
            "reach_m": 0.15,
            "bounding_area_m2": 0.01,
        },
        "source_region": "LEFT_DRAWER",
    },
    {
        "object_id": "short_screw",
        "functions": ["can_fasten"],
        "geometry": {
            "diameter_m": 0.005,
            "length_m": 0.015,
            "recess_profile": "PH2",
            "recess_width_m": 0.005,
            "required_tool_reach_m": 0.025,
            "bounding_area_m2": 0.001,
        },
        "source_region": "LEFT_DRAWER",
    },
    {
        "object_id": "phillips_driver",
        "functions": ["can_drive_screw"],
        "geometry": {
            "tip_profile": "PH2",
            "tip_width_m": 0.004,
            "reach_m": 0.18,
            "bounding_area_m2": 0.012,
        },
        "source_region": "TOOL_CABINET",
    },
    {
        "object_id": "medium_screw",
        "functions": ["can_fasten"],
        "geometry": {
            "diameter_m": 0.005,
            "length_m": 0.045,
            "recess_profile": "PH2",
            "recess_width_m": 0.0045,
            "required_tool_reach_m": 0.025,
            "bounding_area_m2": 0.001,
        },
        "source_region": "TOOL_CABINET",
    },
]

SAMPLE_REGIONS = [
    {
        "region_id": "MAIN_WORKBENCH_ZONE",
        "usable_area_m2": 0.60 * 0.33,
    },
    {
        "region_id": "NARROW_WALL_SHELF",
        "usable_area_m2": 0.010,  # too small for driver + fastener set
    },
    {
        "region_id": "PARTS_TRAY",
        "cavity_volume_m3": 0.002,
        "is_open": True,
    },
    {
        "region_id": "CLOSED_TRAY",
        "cavity_volume_m3": 0.0,
        "is_open": False,
    },
]


class WorkshopSceneTests(unittest.TestCase):
    def test_scene_compiles_with_and_without_google_robot(self):
        for robot in ("none", "google"):
            scene = WorkshopScene(robot=robot, variant="F0_BASE")
            self.assertEqual(scene.has_robot, robot == "google")
            for camera_name in WORKSHOP_CAMERAS:
                cam_id = mujoco.mj_name2id(
                    scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                )
                self.assertGreaterEqual(cam_id, 0, f"Missing camera: {camera_name}")

    def test_storage_objects_revealed_only_after_opening(self):
        scene = WorkshopScene("none", variant="F0_BASE")
        initial_names = dict(scene.get_visible_object_instances())
        self.assertNotIn("workshop_long_phillips_driver", initial_names)
        self.assertNotIn("workshop_power_driver", initial_names)
        self.assertNotIn("workshop_stubby_phillips_driver", initial_names)

        # Open LEFT_DRAWER
        left_revealed = scene.open_container("LEFT_DRAWER")
        self.assertIn("workshop_stubby_phillips_driver", left_revealed)
        self.assertNotIn("workshop_long_phillips_driver", left_revealed)
        self.assertEqual(
            scene.get_instance_source_region("workshop_stubby_phillips_driver"),
            "LEFT_DRAWER",
        )
        self.assertIsNone(
            scene.get_instance_source_region("workshop_long_phillips_driver")
        )

        # Open TOOL_CABINET
        cab_revealed = scene.open_container("TOOL_CABINET")
        self.assertIn("workshop_long_phillips_driver", cab_revealed)
        self.assertEqual(
            scene.get_instance_source_region("workshop_long_phillips_driver"),
            "TOOL_CABINET",
        )

    def test_candidate_work_surfaces_and_parts_containers(self):
        scene = WorkshopScene("none", variant="F0_BASE")
        surfaces = scene.get_candidate_work_surfaces()
        surface_ids = [s["region_id"] for s in surfaces]
        self.assertIn("MAIN_WORKBENCH_ZONE", surface_ids)
        self.assertIn("TOOL_CART_TOP", surface_ids)
        self.assertIn("NARROW_WALL_SHELF", surface_ids)

        containers = scene.get_candidate_parts_containers()
        container_ids = [c["region_id"] for c in containers]
        self.assertIn("PARTS_TRAY", container_ids)
        self.assertIn("HARDWARE_BIN", container_ids)

    def test_joint_seal_is_stable_and_can_be_moved_to_tray(self):
        scene = WorkshopScene("none", variant="F0_BASE")
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
        self.assertEqual(state["joint_seal_location"], "PARTS_TRAY")

    def test_all_three_storage_regions_open_and_close(self):
        scene = WorkshopScene("none", variant="F0_BASE")
        for reg in WORKSHOP_REGIONS:
            scene.open_container(reg)
            self.assertTrue(scene.get_region_observation_states()[reg]["open"])
            scene.close_container(reg)
            self.assertFalse(scene.get_region_observation_states()[reg]["open"])

    def test_all_14_variants_instantiate_cleanly(self):
        config = _load_workshop_variants_config()
        variants = config.get("variants", {})
        self.assertEqual(len(variants), 14, "Expected exactly 14 benchmark variants (F0-F6, I0-I6)")

        for var_name, var_meta in variants.items():
            scene = WorkshopScene(robot="google", variant=var_name)
            self.assertEqual(scene.variant_name, var_name)
            self.assertEqual(
                scene.variant_meta.get("intended_outcome"),
                var_meta.get("intended_outcome"),
            )
            # Verify data is physically valid (no NaNs in qpos or qvel)
            self.assertFalse(
                any(np.isnan(scene.data.qpos)),
                f"NaN found in qpos for variant {var_name}",
            )
            self.assertFalse(
                any(np.isnan(scene.data.qvel)),
                f"NaN found in qvel for variant {var_name}",
            )

    def test_workshop_has_five_views_for_each_observation_stage(self):
        config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
        self.assertEqual(
            config["inspection_sequence"],
            ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
        )
        self.assertEqual(set(config["camera_slots"].values()), set(WORKSHOP_CAMERAS))
        for region in config["regions"].values():
            self.assertEqual(len(region["cameras"]), 5)

    def test_open_drawers_produce_fresh_region_gated_rgbd_evidence(self):
        scene = WorkshopScene("none", variant="F0_BASE")
        checker = GeometryChecker(scene, width=320, height=240)
        initial_run = checker.run_region_inspection("INITIAL")
        self.assertGreater(
            len(initial_run.clouds["workshop_joint_seal"].points), 20
        )
        scene.open_container("LEFT_DRAWER")
        left_run = checker.run_region_inspection("LEFT_DRAWER")
        for instance_name in (
            "workshop_stubby_phillips_driver",
            "workshop_short_phillips_screw",
        ):
            self.assertGreater(len(left_run.clouds[instance_name].points), 20)

        scene.close_container("LEFT_DRAWER")
        scene.open_container("TOOL_CABINET")
        cab_run = checker.run_region_inspection("TOOL_CABINET")
        for instance_name in (
            "workshop_long_phillips_driver",
            "workshop_medium_phillips_screw",
        ):
            self.assertGreater(len(cab_run.clouds[instance_name].points), 20)

    def test_workshop_pointcloud_runner_captures_all_four_stages(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "workshop_run"
            scene, manifest = run_workshop_pointcloud(
                output,
                robot="none",
                variant="F0_BASE",
                width=320,
                height=240,
            )

            self.assertEqual(
                [stage["region_id"] for stage in manifest["stages"]],
                ["INITIAL", "LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
            )
            self.assertEqual(manifest["segmentation"], "oracle")
            self.assertTrue((output / "manifest.json").is_file())
            for stage in manifest["stages"]:
                stage_dir = output / stage["directory"]
                self.assertTrue((stage_dir / "stage_summary.json").is_file())
                self.assertTrue((stage_dir / stage["combined_ply"]).is_file())
                self.assertGreater(stage["total_point_count"], 20)


class WorkshopAlternativeTests(unittest.TestCase):
    def test_joint_reasoning_rejects_near_misses_then_selects_screw(self):
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
            observed_objects=SAMPLE_OBJECTS,
            target_geometry=TARGET,
            ranked_proposals=proposals,
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["selected"]["rank"], 3)
        self.assertFalse(result["evaluated"][0]["geometry_checks"]["fits_hole"])
        self.assertFalse(
            result["evaluated"][1]["geometry_checks"]["reaches_joint"]
        )

    def test_joint_object_and_region_coupling(self):
        proposals = [
            {
                "rank": 1,
                "method": "screw",
                "tool_object_id": "phillips_driver",
                "fastener_object_id": "medium_screw",
                "work_surface_id": "NARROW_WALL_SHELF",  # packing failure
                "parts_container_id": "PARTS_TRAY",
            },
            {
                "rank": 2,
                "method": "screw",
                "tool_object_id": "phillips_driver",
                "fastener_object_id": "medium_screw",
                "work_surface_id": "MAIN_WORKBENCH_ZONE",  # valid
                "parts_container_id": "PARTS_TRAY",        # valid
            },
        ]
        result = evaluate_ranked_alternatives(
            observed_objects=SAMPLE_OBJECTS,
            target_geometry=TARGET,
            ranked_proposals=proposals,
            observed_regions=SAMPLE_REGIONS,
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["selected"]["rank"], 2)
        self.assertFalse(
            result["evaluated"][0]["region_checks"]["fits_work_surface"]
        )
        self.assertTrue(
            result["evaluated"][1]["region_checks"]["fits_work_surface"]
        )
        self.assertTrue(
            result["evaluated"][1]["region_checks"]["fits_parts_container"]
        )

    def test_unobserved_object_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside visible state"):
            evaluate_ranked_alternatives(
                observed_objects=SAMPLE_OBJECTS[:2],
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

    def test_accepts_valid_ranked_alternatives(self):
        proposal = {
            "method": "nail",
            "tool_object_id": "hammer",
            "fastener_object_id": "large_nail",
        }
        with self.assertRaisesRegex(ValueError, "maximum is 5"):
            evaluate_ranked_alternatives(
                observed_objects=SAMPLE_OBJECTS,
                target_geometry=TARGET,
                ranked_proposals=[
                    {**proposal, "rank": rank} for rank in range(1, 7)
                ],
            )


if __name__ == "__main__":
    unittest.main()
