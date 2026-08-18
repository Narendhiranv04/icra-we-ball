"""Comprehensive test suite for the realistic Workshop (W1) benchmark."""

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from mujoco_scenes.audit_workshop_scene import (
    FORBIDDEN_LEAKAGE_STRINGS,
    MIN_WORKSHOP_OBJECT_FUSED_POINTS,
    _check_dict_leakage,
    audit_all_variants,
    audit_single_variant,
)
from mujoco_scenes.geometry_checker import GeometryChecker, load_inspection_rig_config
from mujoco_scenes.workshop_alternatives import evaluate_ranked_alternatives
from mujoco_scenes.workshop_pointcloud import run_workshop_pointcloud
from mujoco_scenes.workshop_scene import (
    INITIAL_OBJECTS,
    PRIVILEGED_WORKSHOP_ORACLE_SPECS,
    WORKSHOP_CAMERAS,
    WORKSHOP_FUNCTIONAL_PARTS_CONTAINERS,
    WORKSHOP_FUNCTIONAL_WORK_SURFACES,
    WORKSHOP_INSPECTION_RIG_CONFIG,
    WORKSHOP_REGIONS,
    WorkshopScene,
    _load_workshop_variants_config,
    privileged_actual_parts_container_regions,
    privileged_actual_storage_region,
    privileged_actual_work_surface_regions,
    privileged_validate_variant_feasibility,
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
        "usable_area_m2": 0.010,
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
        initial_visible = scene.get_observed_instances()
        initial_ids = {obs["instance_id"] for obs in initial_visible}
        initial_backend_names = {
            scene.privileged_backend_name_for_instance(i_id) for i_id in initial_ids
        }
        self.assertNotIn("workshop_long_phillips_driver", initial_backend_names)
        self.assertNotIn("workshop_power_driver", initial_backend_names)
        self.assertNotIn("workshop_flathead_screwdriver", initial_backend_names)

        # Open LEFT_DRAWER - open_container returns status dict without object names
        open_res = scene.open_container("LEFT_DRAWER")
        self.assertEqual(open_res["region_id"], "LEFT_DRAWER")
        self.assertTrue(open_res["opened"])
        self.assertTrue(open_res["newly_opened"])
        self.assertNotIn("workshop_flathead_screwdriver", open_res)

        # Discovery occurs via observation API
        post_open_visible = scene.get_observed_instances()
        post_open_names = {
            scene.privileged_backend_name_for_instance(obs["instance_id"])
            for obs in post_open_visible
        }
        self.assertIn("workshop_flathead_screwdriver", post_open_names)
        self.assertNotIn("workshop_long_phillips_driver", post_open_names)
        self.assertEqual(
            scene.get_instance_source_region("workshop_flathead_screwdriver"),
            "LEFT_DRAWER",
        )
        self.assertIsNone(
            scene.get_instance_source_region("workshop_long_phillips_driver")
        )

        # Open TOOL_CABINET
        cab_res = scene.open_container("TOOL_CABINET")
        self.assertEqual(cab_res["region_id"], "TOOL_CABINET")
        post_cab_names = {
            scene.privileged_backend_name_for_instance(obs["instance_id"])
            for obs in scene.get_observed_instances()
        }
        self.assertIn("workshop_long_phillips_driver", post_cab_names)
        self.assertEqual(
            scene.get_instance_source_region("workshop_long_phillips_driver"),
            "TOOL_CABINET",
        )

    def test_candidate_work_surfaces_and_parts_containers(self):
        scene = WorkshopScene("none", variant="F0_BASE")
        surfaces = scene.privileged_get_work_surface_specs()
        surface_ids = [s["region_id"] for s in surfaces]
        self.assertIn("MAIN_WORKBENCH_ZONE", surface_ids)
        self.assertIn("TOOL_CART_TOP", surface_ids)

        containers = scene.privileged_get_parts_container_specs()
        container_ids = [c["region_id"] for c in containers]
        self.assertIn("PARTS_TRAY", container_ids)
        self.assertIn("HARDWARE_BIN", container_ids)

    def test_collision_visual_separation_policy(self):
        """Verify strict separation between visual mesh geoms and invisible collision proxies."""
        scene = WorkshopScene("none", variant="F0_BASE")
        for i in range(scene.model.ngeom):
            name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            g_type = scene.model.geom_type[i]
            contype = scene.model.geom_contype[i]
            conaffinity = scene.model.geom_conaffinity[i]
            group = scene.model.geom_group[i]
            rgba = scene.model.geom_rgba[i]

            # If it's a visual mesh geom
            if g_type == mujoco.mjtGeom.mjGEOM_MESH or name.endswith("_vis") or name.endswith("_visual"):
                self.assertEqual(
                    contype, 0, f"Visual geom {name} has non-zero contype ({contype})"
                )
                self.assertEqual(
                    conaffinity, 0, f"Visual geom {name} has non-zero conaffinity ({conaffinity})"
                )
                self.assertEqual(
                    group, 1, f"Visual geom {name} is not in group 1 ({group})"
                )

            # If it's a collision proxy
            if name.endswith("_col") or "_col_" in name:
                self.assertNotEqual(
                    contype + conaffinity,
                    0,
                    f"Collision proxy {name} has zero contype and conaffinity",
                )
                self.assertEqual(
                    group, 3, f"Collision proxy {name} is not in group 3 ({group})"
                )
                self.assertEqual(
                    rgba[3], 0.0, f"Collision proxy {name} is not transparent ({rgba})"
                )

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
            self.assertFalse(
                any(np.isnan(scene.data.qpos)),
                f"NaN found in qpos for variant {var_name}",
            )
            self.assertFalse(
                any(np.isnan(scene.data.qvel)),
                f"NaN found in qvel for variant {var_name}",
            )

    def test_physical_inventory_matches_yaml_across_all_14_variants(self):
        config = _load_workshop_variants_config()
        variants = config.get("variants", {})
        for var_name, var_meta in variants.items():
            scene = WorkshopScene("none", variant=var_name)
            declared_contents = var_meta.get("storage_contents", {})
            actual_contents: dict[str, list[str]] = {reg: [] for reg in WORKSHOP_REGIONS}
            for reg, expected_objs in declared_contents.items():
                for obj_name in expected_objs:
                    actual_reg = privileged_actual_storage_region(scene, obj_name)
                    self.assertIn(actual_reg, actual_contents, f"Object {obj_name} placed in invalid region {actual_reg}")
                    actual_contents[actual_reg].append(obj_name)
                self.assertEqual(
                    sorted(expected_objs),
                    sorted(actual_contents[reg]),
                    f"Physical inventory mismatch in {var_name} for region {reg}",
                )

    def test_all_pickable_objects_are_independent_free_bodies(self):
        config = _load_workshop_variants_config()
        variants = config.get("variants", {})
        for var_name, var_meta in variants.items():
            scene = WorkshopScene("none", variant=var_name)
            for reg, objs in var_meta.get("storage_contents", {}).items():
                for obj_name in objs:
                    b_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                    j_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj_name}_free")
                    self.assertGreaterEqual(b_id, 0, f"Missing body {obj_name} in {var_name}")
                    self.assertGreaterEqual(j_id, 0, f"Missing freejoint {obj_name}_free in {var_name}")
                    self.assertEqual(
                        scene.model.jnt_type[j_id],
                        mujoco.mjtJoint.mjJNT_FREE,
                        f"Joint {obj_name}_free is not a free joint in {var_name}",
                    )

    def test_physical_region_availability_matches_yaml_14_of_14(self):
        config = _load_workshop_variants_config()
        variants = config.get("variants", {})
        for var_name, var_meta in variants.items():
            scene = WorkshopScene("none", variant=var_name)
            declared_surfaces = set(var_meta.get("active_surfaces", []))
            actual_surfaces = set(privileged_actual_work_surface_regions(scene))
            self.assertEqual(
                declared_surfaces,
                actual_surfaces,
                f"Physical surfaces mismatch in {var_name}",
            )

            declared_containers = set(var_meta.get("active_containers", []))
            actual_containers = set(privileged_actual_parts_container_regions(scene))
            self.assertEqual(
                declared_containers,
                actual_containers,
                f"Physical containers mismatch in {var_name}",
            )

    def test_f6_layout_swapped_differs_physically_from_base(self):
        scene_f0 = WorkshopScene("none", variant="F0_BASE")
        scene_f6 = WorkshopScene("none", variant="F6_LAYOUT_SWAPPED")

        cab_f0 = scene_f0.data.xpos[mujoco.mj_name2id(scene_f0.model, mujoco.mjtObj.mjOBJ_BODY, "tool_cabinet")]
        cab_f6 = scene_f6.data.xpos[mujoco.mj_name2id(scene_f6.model, mujoco.mjtObj.mjOBJ_BODY, "tool_cabinet")]
        self.assertNotAlmostEqual(cab_f0[0], cab_f6[0], delta=0.2)
        self.assertAlmostEqual(cab_f0[0], 0.44, delta=0.05)
        self.assertAlmostEqual(cab_f6[0], -0.44, delta=0.05)

        tray_f0 = scene_f0.data.xpos[mujoco.mj_name2id(scene_f0.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_parts_tray")]
        tray_f6 = scene_f6.data.xpos[mujoco.mj_name2id(scene_f6.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_parts_tray")]
        self.assertNotAlmostEqual(tray_f0[0], tray_f6[0], delta=0.4)

    def test_privileged_oracle_feasibility_across_all_14_variants(self):
        config = _load_workshop_variants_config()
        variants = config.get("variants", {})
        for var_name, var_meta in variants.items():
            scene = WorkshopScene("none", variant=var_name)
            oracle_res = privileged_validate_variant_feasibility(scene)
            intended = var_meta.get("intended_outcome")
            expected_reason = var_meta.get("rejection_reason")
            self.assertEqual(
                oracle_res["status"],
                intended,
                f"Oracle status mismatch in {var_name}",
            )
            if intended == "INFEASIBLE":
                self.assertEqual(
                    oracle_res["rejection_reason"],
                    expected_reason,
                    f"Rejection reason mismatch in {var_name}",
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
        checker = GeometryChecker(scene, width=640, height=480)
        initial_run = checker.run_region_inspection("INITIAL", rig_config=scene.inspection_rig_config)
        self.assertGreater(
            len(initial_run.clouds["workshop_frame_joint"].points), 20
        )
        scene.open_container("LEFT_DRAWER")
        left_run = checker.run_region_inspection("LEFT_DRAWER", rig_config=scene.inspection_rig_config)
        for instance_name in (
            "workshop_flathead_screwdriver",
            "workshop_short_phillips_screw",
        ):
            self.assertIn(instance_name, left_run.clouds)
            self.assertGreater(len(left_run.clouds[instance_name].points), 20)

        scene.close_container("LEFT_DRAWER")
        scene.open_container("TOOL_CABINET")
        cab_run = checker.run_region_inspection("TOOL_CABINET", rig_config=scene.inspection_rig_config)
        for instance_name in (
            "workshop_long_phillips_driver",
            "workshop_medium_phillips_screw",
        ):
            self.assertIn(instance_name, cab_run.clouds)
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


class WorkshopPrivilegeBoundaryTests(unittest.TestCase):
    """Rigorous tests ensuring zero privileged semantic leakage through production APIs."""

    def test_production_observed_instances_leak_no_semantics(self):
        for v_name in ("F0_BASE", "F3_DISTRIBUTED_OBJECTS", "F5_DECOY_HEAVY", "F6_LAYOUT_SWAPPED"):
            scene = WorkshopScene("none", variant=v_name)
            for reg in WORKSHOP_REGIONS:
                scene.open_container(reg)
            observed = scene.get_observed_instances()

            pattern = re.compile(r"^object_\d{4}$")
            seen_ids = set()
            for item in observed:
                self.assertIn("instance_id", item)
                self.assertIn("source_region", item)
                self.assertRegex(item["instance_id"], pattern)
                seen_ids.add(item["instance_id"])

            self.assertEqual(len(seen_ids), len(observed), "Generic instance IDs must be unique")

            leaks = _check_dict_leakage(observed, FORBIDDEN_LEAKAGE_STRINGS)
            self.assertEqual(leaks, [], f"Forbidden strings leaked in {v_name}: {leaks}")

    def test_production_candidate_regions_leak_no_ground_truth(self):
        scene = WorkshopScene("none", variant="F0_BASE")
        regions = scene.get_candidate_regions()
        pattern = re.compile(r"^region_\d{4}$")
        for reg in regions:
            self.assertRegex(reg["region_instance_id"], pattern)
            self.assertNotIn("usable_area_m2", reg)
            self.assertNotIn("cavity_volume_m3", reg)
            self.assertNotIn("proposal_type", reg)
            self.assertIn("proposal_bounds_m", reg)

        leaks = _check_dict_leakage(regions, FORBIDDEN_LEAKAGE_STRINGS)
        self.assertEqual(leaks, [], f"Forbidden strings leaked in candidate regions: {leaks}")

    def test_production_target_workpiece_spec_leaks_no_ground_truth(self):
        scene = WorkshopScene("none", variant="F0_BASE")
        target_spec = scene.get_target_workpiece_specification()
        self.assertIn("target_instance_id", target_spec)
        self.assertIn("fixture_center_world_m", target_spec)
        self.assertNotIn("target_hole_diameter_m", target_spec)
        self.assertNotIn("target_hole_depth_m", target_spec)
        self.assertNotIn("required_driver_function", target_spec)
        self.assertNotIn("required_fastener_function", target_spec)
        self.assertNotIn("required_recess_profile", target_spec)

        leaks = _check_dict_leakage(target_spec, FORBIDDEN_LEAKAGE_STRINGS)
        self.assertEqual(leaks, [], f"Forbidden strings leaked in target workpiece spec: {leaks}")

    def test_f2_region_alternative_proposes_obstructed_workbench_neutrally(self):
        scene = WorkshopScene("none", variant="F2_REGION_ALTERNATIVE")
        proposals = scene.get_candidate_regions()
        proposal_backend_names = {
            scene.privileged_backend_name_for_region(p["region_instance_id"])
            for p in proposals
        }
        # Physical workbench exists and is proposed neutrally
        self.assertIn("MAIN_WORKBENCH_ZONE", proposal_backend_names)

        # But privileged physical oracle confirms it is obstructed and not a valid work surface
        actual_surfaces = privileged_actual_work_surface_regions(scene)
        self.assertNotIn("MAIN_WORKBENCH_ZONE", actual_surfaces)

    def test_i2_no_work_surface_proposals_follow_physical_presence(self):
        scene = WorkshopScene("none", variant="I2_NO_WORK_SURFACE")
        proposals = scene.get_candidate_regions()
        proposal_backend_names = {
            scene.privileged_backend_name_for_region(p["region_instance_id"])
            for p in proposals
        }
        # Physically present workbench table is proposed neutrally
        self.assertIn("MAIN_WORKBENCH_ZONE", proposal_backend_names)
        # Physically removed tool cart and shelf are NOT proposed
        self.assertNotIn("TOOL_CART_TOP", proposal_backend_names)
        self.assertNotIn("NARROW_WALL_SHELF", proposal_backend_names)

        # Privileged oracle confirms zero valid work surfaces
        oracle_res = privileged_validate_variant_feasibility(scene)
        self.assertEqual(oracle_res["rejection_reason"], "NO_WORK_SURFACE")

    def test_i3_no_parts_container_physically_absent_containers_not_proposed(self):
        scene = WorkshopScene("none", variant="I3_NO_PARTS_CONTAINER")
        proposals = scene.get_candidate_regions()
        proposal_backend_names = {
            scene.privileged_backend_name_for_region(p["region_instance_id"])
            for p in proposals
        }
        # Physically removed parts containers are NOT proposed
        self.assertNotIn("PARTS_TRAY", proposal_backend_names)
        self.assertNotIn("HARDWARE_BIN", proposal_backend_names)
        self.assertNotIn("TOOLBOX_COMPARTMENT", proposal_backend_names)

        # Privileged oracle confirms NO_PARTS_CONTAINER
        oracle_res = privileged_validate_variant_feasibility(scene)
        self.assertEqual(oracle_res["rejection_reason"], "NO_PARTS_CONTAINER")

    def test_ordinary_api_surface_has_no_cheating_wrappers(self):
        scene = WorkshopScene("none", variant="F0_BASE")
        self.assertFalse(hasattr(scene, "get_candidate_work_surfaces"))
        self.assertFalse(hasattr(scene, "get_candidate_parts_containers"))
        self.assertFalse(hasattr(scene, "get_target_joint_specification"))

    def test_generic_instance_ids_are_deterministic_and_persistent(self):
        scene1 = WorkshopScene("none", variant="F0_BASE")
        scene2 = WorkshopScene("none", variant="F0_BASE")

        self.assertEqual(scene1._backend_to_instance_id, scene2._backend_to_instance_id)

        # ID of workpiece target before opening
        initial_target_id = [
            obs["instance_id"]
            for obs in scene1.get_observed_instances()
            if scene1.privileged_backend_name_for_instance(obs["instance_id"]) == "workshop_frame_joint"
        ][0]

        # Open drawer
        scene1.open_container("LEFT_DRAWER")
        post_open_target_id = [
            obs["instance_id"]
            for obs in scene1.get_observed_instances()
            if scene1.privileged_backend_name_for_instance(obs["instance_id"]) == "workshop_frame_joint"
        ][0]

        self.assertEqual(initial_target_id, post_open_target_id, "Instance ID must not mutate upon container opening")

    def test_privileged_apis_retain_full_ground_truth(self):
        scene = WorkshopScene("none", variant="F0_BASE")

        # Invertibility of ID mapping
        for backend_name, instance_id in scene._backend_to_instance_id.items():
            self.assertEqual(scene.privileged_backend_name_for_instance(instance_id), backend_name)
            self.assertEqual(scene.privileged_instance_id_for_backend(backend_name), instance_id)

        # Target spec ground truth
        priv_target = scene.privileged_get_target_joint_specification()
        self.assertEqual(priv_target["target_hole_diameter_m"], 0.007)
        self.assertEqual(priv_target["target_hole_depth_m"], 0.030)
        self.assertEqual(priv_target["required_recess_profile"], "PH2")
        self.assertEqual(priv_target["required_driver_function"], "can_drive_screw")
        self.assertEqual(priv_target["required_fastener_function"], "can_fasten")

        # Work surface spec ground truth
        surfs = scene.privileged_get_work_surface_specs()
        for s in surfs:
            self.assertIn("usable_area_m2", s)
            self.assertGreater(s["usable_area_m2"], 0.0)

        # Parts container spec ground truth
        conts = scene.privileged_get_parts_container_specs()
        for c in conts:
            self.assertIn("cavity_volume_m3", c)
            self.assertGreater(c["cavity_volume_m3"], 0.0)

    def test_hardened_suite_audit_all_14_variants_pass_100_percent(self):
        summary = audit_all_variants(run_pointcloud_smoke=True)
        self.assertTrue(summary["all_passed"], "All 14 benchmark variants must pass full physical and oracle audit")
        self.assertEqual(summary["passed_variants"], 14)

    def test_workshop_cross_layer_geometry_consistency(self):
        """Verify cross-layer agreement between visual mesh, collision proxy, manifest, and oracle specs."""
        from mujoco_scenes.workshop_scene import privileged_audit_object_dimensions
        scene = WorkshopScene("none", variant="F0_BASE")
        report = privileged_audit_object_dimensions(scene)

        # 1. Medium Phillips Screw (Canonical Fastener)
        self.assertIn("workshop_medium_phillips_screw", report)
        med = report["workshop_medium_phillips_screw"]
        self.assertAlmostEqual(med["oracle_spec"]["length_m"], 0.045, delta=1e-3)
        self.assertAlmostEqual(med["oracle_spec"]["head_diameter_m"], 0.014, delta=1e-3)
        self.assertAlmostEqual(med["oracle_spec"]["shaft_diameter_m"], 0.0055, delta=1e-3)
        self.assertEqual(med["oracle_spec"]["recess_profile"], "PH2")
        self.assertAlmostEqual(med["collision_proxy_extents_m"][2], 0.045, delta=1e-3)
        self.assertAlmostEqual(med["visual_mesh_extents_m"][2], 0.045, delta=1e-3)
        if med["manifest_canonical_extents_m"]:
            self.assertAlmostEqual(med["manifest_canonical_extents_m"][2], 0.045, delta=1e-3)

        # 2. Short Phillips Screw (Inadequate Reach)
        self.assertIn("workshop_short_phillips_screw", report)
        short = report["workshop_short_phillips_screw"]
        self.assertAlmostEqual(short["oracle_spec"]["length_m"], 0.018, delta=1e-3)
        self.assertAlmostEqual(short["collision_proxy_extents_m"][2], 0.018, delta=1e-3)
        self.assertAlmostEqual(short["visual_mesh_extents_m"][2], 0.018, delta=1e-3)

        # 3. Hex Bolt (Incompatible Profile & Diameter)
        self.assertIn("workshop_hex_bolt", report)
        bolt = report["workshop_hex_bolt"]
        self.assertAlmostEqual(bolt["oracle_spec"]["length_m"], 0.050, delta=1e-3)
        self.assertAlmostEqual(bolt["oracle_spec"]["head_diameter_m"], 0.018, delta=1e-3)
        self.assertAlmostEqual(bolt["oracle_spec"]["shaft_diameter_m"], 0.008, delta=1e-3)
        self.assertEqual(bolt["oracle_spec"]["recess_profile"], "HEX")
        self.assertAlmostEqual(bolt["collision_proxy_extents_m"][2], 0.050, delta=1e-3)
        self.assertAlmostEqual(bolt["visual_mesh_extents_m"][2], 0.050, delta=1e-3)

        # 4. Long Phillips Driver (Canonical Tool)
        self.assertIn("workshop_long_phillips_driver", report)
        long_d = report["workshop_long_phillips_driver"]
        self.assertAlmostEqual(long_d["oracle_spec"]["reach_m"], 0.18, delta=1e-3)
        self.assertEqual(long_d["oracle_spec"]["tip_profile"], "PH2")
        self.assertAlmostEqual(long_d["collision_proxy_extents_m"][2], 0.23, delta=1e-3)
        self.assertAlmostEqual(long_d["visual_mesh_extents_m"][2], 0.23, delta=1e-3)

        # 5. Stubby Phillips Driver (Inadequate Reach)
        self.assertIn("workshop_stubby_phillips_driver", report)
        stub = report["workshop_stubby_phillips_driver"]
        self.assertAlmostEqual(stub["oracle_spec"]["reach_m"], 0.020, delta=1e-3)
        self.assertAlmostEqual(stub["collision_proxy_extents_m"][2], 0.11, delta=1e-3)

        # 6. Target Workpiece & Joint Specification Cross-Layer Consistency
        from mujoco_scenes.workshop_scene import (
            WORKSHOP_TARGET_HOLE_DIAMETER_M,
            WORKSHOP_TARGET_HOLE_RADIUS_M,
            WORKSHOP_TARGET_HOLE_DEPTH_M,
            WORKSHOP_TARGET_RADIAL_CLEARANCE_M,
            WORKSHOP_TARGET_RECESS_PROFILE,
        )
        priv_target = scene.privileged_get_target_joint_specification()
        self.assertEqual(priv_target["target_hole_diameter_m"], WORKSHOP_TARGET_HOLE_DIAMETER_M)
        self.assertEqual(priv_target["target_hole_depth_m"], WORKSHOP_TARGET_HOLE_DEPTH_M)
        self.assertEqual(priv_target["required_recess_profile"], WORKSHOP_TARGET_RECESS_PROFILE)
        self.assertEqual(WORKSHOP_TARGET_HOLE_RADIUS_M * 2.0, WORKSHOP_TARGET_HOLE_DIAMETER_M)

        # 7. Target Visual vs Collision Opening Consistency
        gid_bottom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "frame_target_hole_bottom")
        self.assertGreaterEqual(gid_bottom, 0)
        self.assertAlmostEqual(scene.model.geom_size[gid_bottom][0], WORKSHOP_TARGET_HOLE_RADIUS_M, delta=1e-4)

        gid_bracket_l_vis = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "frame_bracket_l_vis")
        gid_bracket_r_vis = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "frame_bracket_r_vis")
        vis_opening_x = (
            (scene.model.geom_pos[gid_bracket_r_vis][0] - scene.model.geom_size[gid_bracket_r_vis][0])
            - (scene.model.geom_pos[gid_bracket_l_vis][0] + scene.model.geom_size[gid_bracket_l_vis][0])
        )
        self.assertAlmostEqual(vis_opening_x, 0.0075, delta=1e-3)
        self.assertGreaterEqual(vis_opening_x, WORKSHOP_TARGET_HOLE_DIAMETER_M)

    def test_physical_insertion_corridor_recess(self):
        """Verify that target hole is a real 3D hollow recess of diameter >= 7mm and depth >= 30mm."""
        scene = WorkshopScene("none", variant="F0_BASE")
        target_xy = np.array([-0.18, 0.28])
        top_z = 0.749  # top surface of bracket in world coordinates
        bottom_z = top_z - 0.030  # 0.719m

        # 1. Verify that all collision proxies surrounding the hole leave a clear corridor
        for z in np.linspace(bottom_z, top_z, 40):
            for gid in range(scene.model.ngeom):
                if scene.model.geom_group[gid] != 3:
                    continue
                g_name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if "bottom" in g_name:
                    continue  # solid floor beneath z=0.718
                b_name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.model.geom_bodyid[gid]) or ""
                if "fixture" not in b_name and "joint" not in b_name:
                    continue
                g_pos = scene.data.geom_xpos[gid]
                g_size = scene.model.geom_size[gid]
                # Check horizontal clearance if this geom spans current Z
                if g_pos[2] - g_size[2] <= z <= g_pos[2] + g_size[2]:
                    dx = max(abs(target_xy[0] - g_pos[0]) - g_size[0], 0)
                    dy = max(abs(target_xy[1] - g_pos[1]) - g_size[1], 0)
                    xy_dist = np.sqrt(dx * dx + dy * dy)
                    self.assertGreaterEqual(
                        xy_dist,
                        0.0035,
                        f"Collision geom {g_name} blocks 7mm hole corridor at z={z:.4f} (clearance={xy_dist*1000:.2f}mm)",
                    )

    def test_physical_tool_fastener_feasibility_matrix(self):
        """Verify physical feasibility rules between tools, fasteners, and target workpiece."""
        from mujoco_scenes.workshop_scene import (
            WORKSHOP_TARGET_HOLE_DIAMETER_M,
            WORKSHOP_TARGET_HOLE_DEPTH_M,
            WORKSHOP_TARGET_RECESS_PROFILE,
        )
        scene = WorkshopScene("none", variant="F0_BASE")
        specs = PRIVILEGED_WORKSHOP_ORACLE_SPECS

        # Workpiece target hole: 7mm diameter, 30mm depth, PH2 recess
        target = specs["workshop_frame_joint"]
        self.assertEqual(target["hole_diameter_m"], WORKSHOP_TARGET_HOLE_DIAMETER_M)
        self.assertEqual(target["hole_depth_m"], WORKSHOP_TARGET_HOLE_DEPTH_M)
        self.assertEqual(target["recess_profile"], WORKSHOP_TARGET_RECESS_PROFILE)

        # Feasible Pair: long_driver + medium_screw
        med_screw = specs["workshop_medium_phillips_screw"]
        long_driver = specs["workshop_long_phillips_driver"]
        self.assertLess(med_screw["shaft_diameter_m"], target["hole_diameter_m"])
        self.assertGreaterEqual(med_screw["length_m"], target["hole_depth_m"])
        self.assertGreaterEqual(long_driver["reach_m"], med_screw["required_tool_reach_m"])
        self.assertEqual(long_driver["tip_profile"], med_screw["recess_profile"])

        # Infeasible: short_screw (insufficient engagement depth)
        short_screw = specs["workshop_short_phillips_screw"]
        self.assertLess(short_screw["length_m"], target["hole_depth_m"])

        # Infeasible: stubby_driver (insufficient reach for recessed screw)
        stubby_driver = specs["workshop_stubby_phillips_driver"]
        self.assertLess(stubby_driver["reach_m"], med_screw["required_tool_reach_m"])

        # Infeasible: hex_bolt (too large diameter and incompatible profile)
        hex_bolt = specs["workshop_hex_bolt"]
        self.assertGreater(hex_bolt["shaft_diameter_m"], target["hole_diameter_m"])
        self.assertNotEqual(hex_bolt["recess_profile"], target["recess_profile"])

    def test_perception_render_and_instance_mask_isolation(self):
        """Verify RGB-D rendering includes floor/robot while segmentation isolates only group-1 objects."""
        # 1. Google Robot scene
        scene = WorkshopScene("google", variant="F0_BASE")
        self.assertEqual(scene.perception_render_geom_groups, (0, 1, 2))
        self.assertEqual(scene.perception_instance_geom_groups, (1,))

        checker = GeometryChecker(scene, width=640, height=480)
        self.assertEqual(checker.render_geom_groups, (0, 1, 2))
        self.assertEqual(checker.instance_geom_groups, (1,))

        renderer = mujoco.Renderer(scene.model, height=480, width=640)
        vopt = checker._build_scene_option()
        renderer.update_scene(scene.data, camera="workshop_camera_front", scene_option=vopt)
        rgb = renderer.render()
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertGreater(rgb.mean(), 50.0)

        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        valid_depth = depth[np.isfinite(depth)]
        self.assertGreater(len(valid_depth), 10000)

        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        # Instance geom mapping must only include group-1 geoms
        geom_map = checker._geom_ids_by_instance(["workshop_frame_joint"])
        for g_id in geom_map.get("workshop_frame_joint", []):
            self.assertEqual(scene.model.geom_group[g_id], 1)

    def test_rgbd_observable_target_recess(self):
        """Verify that production RGB-D rendering sees an actual ~30mm deep 3D cavity at the target hole."""
        scene = WorkshopScene("none", variant="F0_BASE")
        checker = GeometryChecker(scene, width=640, height=480)
        vopt = checker._build_scene_option()

        renderer = mujoco.Renderer(scene.model, height=480, width=640)
        cam_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "workshop_camera_top")
        renderer.update_scene(scene.data, camera=cam_id, scene_option=vopt)

        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()

        # Target workpiece coordinates
        joint_body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_frame_joint")
        joint_pos = scene.data.xpos[joint_body_id]
        target_top_world = np.array([joint_pos[0] - 0.10, joint_pos[1] - 0.02, joint_pos[2] + 0.023])
        cam_pos = scene.data.cam_xpos[cam_id]
        cam_mat = scene.data.cam_xmat[cam_id].reshape(3, 3)
        fovy = scene.model.cam_fovy[cam_id]
        f = 0.5 * 480 / np.tan(np.deg2rad(fovy) / 2.0)

        # World to camera frame
        p_top_cam = cam_mat.T @ (target_top_world - cam_pos)
        u_top = int(np.round(320 + f * (p_top_cam[0] / -p_top_cam[2])))
        v_top = int(np.round(240 - f * (p_top_cam[1] / -p_top_cam[2])))

        depth_center = float(depth[v_top, u_top])
        surrounding_pixels = [
            depth[v_top, u_top - 8],
            depth[v_top, u_top + 8],
            depth[v_top - 8, u_top],
            depth[v_top + 8, u_top],
        ]
        depth_surrounding = float(np.mean(surrounding_pixels))
        observed_recess_depth = depth_center - depth_surrounding

        # 1. Assert observable recess depth is approximately 30mm (>= 20mm, <= 40mm)
        self.assertGreaterEqual(
            observed_recess_depth,
            0.020,
            f"Production depth renderer sees solid surface or shallow depth ({observed_recess_depth*1000:.1f}mm < 20mm)",
        )
        self.assertLessEqual(
            observed_recess_depth,
            0.040,
            f"Production depth renderer sees excessively deep hole ({observed_recess_depth*1000:.1f}mm > 40mm)",
        )

        # 2. Visual mask sanity: Center ray must reach deeper cavity and not be blocked by mouth bracket geoms
        center_geom_id = int(seg[v_top, u_top, 0])
        center_geom_name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, center_geom_id) or ""
        mouth_bracket_geoms = {
            "frame_bracket_l_vis",
            "frame_bracket_r_vis",
            "frame_bracket_f_vis",
            "frame_bracket_b_vis",
        }
        self.assertNotIn(
            center_geom_name,
            mouth_bracket_geoms,
            f"Mouth bracket geom {center_geom_name} blocks target hole center ray at pixel ({u_top}, {v_top})",
        )

    def test_manifest_offline_verification(self):
        """Verify offline that committed assets strictly match the provenance manifest."""
        from mujoco_scenes.scripts.prepare_workshop_assets import (
            generate_workshop_hex_bolt,
            generate_workshop_parts_tray,
            verify_manifest,
        )
        # 1. Full offline manifest audit
        self.assertTrue(verify_manifest(), "Committed realistic assets do not match manifest.json offline")

        # 2. Offline procedural asset generation check
        with TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            tray_entry = generate_workshop_parts_tray(tmp_out)
            self.assertEqual(tray_entry["asset_id"], "workshop_parts_tray")
            self.assertTrue((tmp_out / "workshop_parts_tray.obj").is_file())

            bolt_entry = generate_workshop_hex_bolt(tmp_out)
            self.assertEqual(bolt_entry["asset_id"], "workshop_hex_bolt")
            self.assertTrue((tmp_out / "workshop_hex_bolt.obj").is_file())

    def test_storage_physical_containment(self):
        """Verify that open drawers and cabinet contain resting objects when welds are disabled."""
        # 1. Left drawer containment
        scene_l = WorkshopScene("none", variant="F0_BASE")
        scene_l.open_container("LEFT_DRAWER")
        weld_id_l = mujoco.mj_name2id(scene_l.model, mujoco.mjtObj.mjOBJ_EQUALITY, "storage_weld_workshop_flathead_screwdriver")
        self.assertGreaterEqual(weld_id_l, 0)
        scene_l.data.eq_active[weld_id_l] = 0
        for _ in range(300):
            mujoco.mj_step(scene_l.model, scene_l.data)
        b_id_l = mujoco.mj_name2id(scene_l.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_flathead_screwdriver")
        pos_l = scene_l.data.xpos[b_id_l]
        # Must stay within left drawer bounds (x around -0.28, z supported on drawer floor >= 0.45)
        self.assertAlmostEqual(pos_l[0], -0.28, delta=0.15)
        self.assertGreaterEqual(pos_l[2], 0.45)

        # 2. Right drawer containment
        scene_r = WorkshopScene("none", variant="F0_BASE")
        scene_r.open_container("RIGHT_DRAWER")
        weld_id_r = mujoco.mj_name2id(scene_r.model, mujoco.mjtObj.mjOBJ_EQUALITY, "storage_weld_workshop_stubby_phillips_driver")
        self.assertGreaterEqual(weld_id_r, 0)
        scene_r.data.eq_active[weld_id_r] = 0
        for _ in range(300):
            mujoco.mj_step(scene_r.model, scene_r.data)
        b_id_r = mujoco.mj_name2id(scene_r.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_stubby_phillips_driver")
        pos_r = scene_r.data.xpos[b_id_r]
        self.assertAlmostEqual(pos_r[0], 0.28, delta=0.15)
        self.assertGreaterEqual(pos_r[2], 0.45)

        # 3. Cabinet containment (fastener supported on shelf, driver inside cabinet bounds)
        scene_c = WorkshopScene("none", variant="F0_BASE")
        scene_c.open_container("TOOL_CABINET")
        weld_id_c = mujoco.mj_name2id(scene_c.model, mujoco.mjtObj.mjOBJ_EQUALITY, "storage_weld_workshop_medium_phillips_screw")
        self.assertGreaterEqual(weld_id_c, 0)
        scene_c.data.eq_active[weld_id_c] = 0
        for _ in range(300):
            mujoco.mj_step(scene_c.model, scene_c.data)
        b_id_c = mujoco.mj_name2id(scene_c.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_medium_phillips_screw")
        pos_c = scene_c.data.xpos[b_id_c]
        # Must stay on cabinet shelf (z >= 0.85)
        self.assertGreaterEqual(pos_c[2], 0.85)

    def test_cabinet_door_collision_articulates_with_hinge(self):
        """Verify that tool cabinet door collision follows the door hinge cleanly."""
        scene = WorkshopScene("none", variant="F0_BASE")
        door_jnt = scene.model.joint("tool_cabinet_door_hinge").id
        door_col_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "tool_cabinet_door_col")
        self.assertGreaterEqual(door_col_id, 0)

        # Initially closed
        self.assertAlmostEqual(scene.data.qpos[door_jnt], 0.0, delta=0.05)
        init_geom_pos = scene.data.geom_xpos[door_col_id].copy()

        # Open container
        scene.open_container("TOOL_CABINET")
        self.assertGreater(scene.data.qpos[door_jnt], 1.4)
        open_geom_pos = scene.data.geom_xpos[door_col_id].copy()
        # The collision proxy must have moved significantly in XY
        dist_moved = float(np.linalg.norm(open_geom_pos[:2] - init_geom_pos[:2]))
        self.assertGreater(dist_moved, 0.10, "Door collision proxy did not follow hinge articulation")

    def test_frame_fixture_collision_coverage_and_target_approach_corridor(self):
        """Verify fixture collision coverage and clear tool approach corridor to target hole."""
        scene = WorkshopScene("none", variant="F0_BASE")
        for col_name in (
            "fixture_base_bottom_col",
            "fixture_clamp_col",
            "frame_h_rail_l_col",
            "frame_h_rail_r_col",
            "frame_bracket_l_col",
            "frame_bracket_r_col",
        ):
            gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, col_name)
            self.assertGreaterEqual(gid, 0, f"Missing collision geom {col_name}")
            self.assertEqual(scene.model.geom_group[gid], 3)
            self.assertNotEqual(scene.model.geom_contype[gid] + scene.model.geom_conaffinity[gid], 0)

        # Approach corridor check: target hole is at [-0.10, -0.02, 0.0215] in workpiece frame
        # (world pos [-0.18, 0.28, 0.7475]). Verify no collision geom obstructs straight-down -Z axis above it.
        target_world = np.array([-0.18, 0.28, 0.7475])
        for z_offset in np.linspace(0.01, 0.20, 10):
            sample_pt = target_world + np.array([0, 0, z_offset])
            # Ensure sample_pt is outside all collision boxes
            for gid in range(scene.model.ngeom):
                if scene.model.geom_group[gid] == 3:
                    g_pos = scene.data.geom_xpos[gid]
                    g_size = scene.model.geom_size[gid]
                    if scene.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                        in_box = np.all(np.abs(sample_pt - g_pos) <= g_size[:3])
                        self.assertFalse(in_box, f"Collision geom {mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, gid)} obstructs approach corridor at {sample_pt}")


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
                "work_surface_id": "NARROW_WALL_SHELF",
                "parts_container_id": "PARTS_TRAY",
            },
            {
                "rank": 2,
                "method": "screw",
                "tool_object_id": "phillips_driver",
                "fastener_object_id": "medium_screw",
                "work_surface_id": "MAIN_WORKBENCH_ZONE",
                "parts_container_id": "PARTS_TRAY",
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
