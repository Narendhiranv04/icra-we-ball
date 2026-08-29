"""Lightweight integrity checks for the separate Workshop L five-view profile."""

from pathlib import Path
import unittest

import yaml

from mujoco_scenes.workshop_phase1.capture import ProductionInspectionCapture
from mujoco_scenes.workshop_phase1.requirements import ManualWorkshopFMContract


ROOT = Path(__file__).resolve().parents[2]
RIG_PATH = ROOT / "mujoco_scenes/configs/workshop_inspection_rigs_yoloworld_l_five_view_close.yaml"
PHASE_PATH = ROOT / "mujoco_scenes/configs/workshop_phase1_yoloworld_l_five_view_close.yaml"
CONTRACT_PATH = ROOT / "mujoco_scenes/configs/workshop_phase1_fm_contract.yaml"
VISUAL_PATH = ROOT / "mujoco_scenes/configs/workshop_visual_profile_yoloworld_l.yaml"
GEOMETRY_PATH = ROOT / "mujoco_scenes/configs/workshop_geometry_inference_yoloworld_l.yaml"


class TestWorkshopYoloWorldLFiveViewConfig(unittest.TestCase):
    def test_separate_rig_has_five_complete_views_per_stage(self):
        capture = ProductionInspectionCapture(rig_config_path=RIG_PATH)
        config = capture._configuration
        roles = {"LEFT", "RIGHT", "TOP", "FRONT", "CLOSE"}
        self.assertEqual(set(config["camera_slots"]), roles)
        self.assertEqual(
            set(config["regions"]),
            {"INITIAL", "LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"},
        )
        for region in config["regions"].values():
            self.assertEqual(set(region["cameras"]), roles)

    def test_phase_profile_selects_l_and_full_prediction_artifacts(self):
        config = yaml.safe_load(PHASE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["perception"]["detector"]["checkpoint"], "yolov8l-worldv2.pt")
        self.assertEqual(config["pipeline"]["inspection_rig_config_path"], str(RIG_PATH.relative_to(ROOT)))
        self.assertEqual(config["pipeline"]["visual_profile_path"], str(VISUAL_PATH.relative_to(ROOT)))
        self.assertEqual(config["pipeline"]["geometry_config_path"], str(GEOMETRY_PATH.relative_to(ROOT)))
        self.assertIn("cordless power drill", config["perception"]["detector"]["supplemental_prompts"])
        self.assertIn("Phillips head screw", config["perception"]["detector"]["supplemental_prompts"])
        self.assertTrue(config["inspection"]["early_stop"])
        self.assertTrue(config["artifacts"]["save_all_detection_overlays"])
        self.assertTrue(config["artifacts"]["save_all_bbox_predictions"])
        self.assertTrue(config["artifacts"]["save_raw_rgb"])
        self.assertEqual(
            config["tracking"]["stage_object_merge_distance_threshold_m"], 0.015
        )

    def test_visual_profile_is_bright_shadow_free_and_limits_driver_override(self):
        profile = yaml.safe_load(VISUAL_PATH.read_text(encoding="utf-8"))
        self.assertTrue(profile["lighting"]["disable_cast_shadows"])
        self.assertGreaterEqual(min(profile["lighting"]["headlight"]["ambient"]), 0.33)
        self.assertIn("cabinet_interior_mat", profile["materials"])
        driver_names = {
            "workshop_long_phillips_driver_vis",
            "workshop_power_driver_vis",
        }
        self.assertTrue(driver_names.isdisjoint(profile.get("geoms", {})))
        self.assertEqual(
            profile["geom_prefixes"]["workshop_medium_phillips_screw_profile_"]["rgba"],
            [0.24, 0.27, 0.31, 1.0],
        )
        self.assertEqual(
            profile["geom_prefixes"]["workshop_medium_phillips_screw_profile_"]["group"],
            1,
        )

    def test_l_contract_covers_every_benchmark_category(self):
        contract = ManualWorkshopFMContract(CONTRACT_PATH)
        entries = contract.get_ranked_detector_vocabulary()
        self.assertEqual(
            {entry["canonical_label"] for entry in entries},
            {"screwdriver", "power_driver", "screw", "hammer"},
        )
        self.assertEqual(len(contract.get_detector_prompts()), 4)

    def test_l_geometry_rejects_oversized_driver_fragments(self):
        geometry = yaml.safe_load(GEOMETRY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(geometry["relations"]["grasp_allowance_m"], 0.040)
        self.assertEqual(geometry["relations"]["maximum_driver_length_m"], 0.250)
        self.assertEqual(
            geometry["relations"]["maximum_driver_cross_section_m"], 0.100
        )
        self.assertEqual(
            geometry["relations"]["slender_driver_max_cross_section_ratio"], 0.30
        )
        self.assertEqual(
            geometry["relations"]["compact_driver_min_cross_section_ratio"], 0.55
        )
        self.assertEqual(
            geometry["relations"]["compact_driver_minimum_cross_section_m"], 0.065
        )
        self.assertEqual(
            geometry["relations"]["compact_driver_maximum_cross_section_m"], 0.220
        )
        self.assertEqual(
            geometry["relations"]["compact_driver_maximum_length_m"], 0.250
        )
        self.assertEqual(
            geometry["relations"]["compact_driver_partial_maximum_length_m"], 0.125
        )
        self.assertEqual(
            geometry["relations"]["compact_driver_full_minimum_length_m"], 0.150
        )
        self.assertEqual(
            geometry["relations"]["maximum_fastener_cross_section_m"], 0.043
        )
        self.assertEqual(geometry["measurement"]["slot_anisotropy_ratio"], 4.5)
        self.assertEqual(geometry["measurement"]["slot_min_transverse_length_m"], 0.006)
        self.assertEqual(geometry["measurement"]["slot_max_transverse_length_m"], 0.015)
        self.assertEqual(geometry["measurement"]["hex_radial_symmetry_ratio"], 1.25)
        self.assertEqual(geometry["measurement"]["minimum_interface_points"], 5)
        self.assertEqual(geometry["relations"]["minimum_fastener_camera_count"], 2)
        self.assertEqual(geometry["relations"]["target_length_tolerance_m"], 0.0061)
        self.assertEqual(
            geometry["region_semantic_association"]["minimum_padding_world_m"],
            [0.0, 0.15, 0.25],
        )


if __name__ == "__main__":
    unittest.main()
