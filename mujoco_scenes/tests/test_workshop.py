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
        self.assertAlmostEqual(cab_f0[0], 0.38, delta=0.05)
        self.assertAlmostEqual(cab_f6[0], -0.38, delta=0.05)

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

    def test_asset_generation_reproducibility(self):
        """Verify that prepare_workshop_assets reproduces exact committed mesh geometries."""
        from mujoco_scenes.scripts.prepare_workshop_assets import prepare_assets
        import trimesh

        with TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir) / "assets"
            manifest = prepare_assets(output_dir=tmp_out)
            self.assertGreaterEqual(len(manifest["assets"]), 12)

            committed_dir = Path(__file__).resolve().parents[1] / "assets" / "workshop_realistic"
            critical_assets = [
                "workshop_medium_phillips_screw.obj",
                "workshop_short_phillips_screw.obj",
                "workshop_parts_tray.obj",
                "workshop_long_phillips_driver.obj",
                "workshop_stubby_phillips_driver.obj",
                "workshop_hex_bolt.obj",
            ]
            for asset_name in critical_assets:
                gen_p = tmp_out / asset_name
                comm_p = committed_dir / asset_name
                self.assertTrue(gen_p.is_file(), f"Missing generated {asset_name}")
                self.assertTrue(comm_p.is_file(), f"Missing committed {asset_name}")

                m_gen = trimesh.load(gen_p)
                m_comm = trimesh.load(comm_p)
                self.assertEqual(len(m_gen.vertices), len(m_comm.vertices), f"Vertex count mismatch in {asset_name}")
                self.assertEqual(len(m_gen.faces), len(m_comm.faces), f"Face count mismatch in {asset_name}")
                max_diff = float(np.max(np.abs(m_gen.extents - m_comm.extents)))
                self.assertLess(max_diff, 1e-5, f"Dimension mismatch in {asset_name}: {max_diff}")

    def test_manifest_asset_completeness_and_truthfulness(self):
        """Verify that manifest.json accurately reflects all assets and their provenance."""
        import json
        manifest_path = Path(__file__).resolve().parents[1] / "assets" / "workshop_realistic" / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIn("assets", manifest)
        assets_by_id = {a["asset_id"]: a for a in manifest["assets"]}
        self.assertIn("workshop_parts_tray", assets_by_id)
        tray_meta = assets_by_id["workshop_parts_tray"]
        self.assertEqual(tray_meta["source"], "project-generated procedural mesh")
        self.assertEqual(tray_meta["license"], "CC0-1.0")

        # Verify dimensions and files
        for asset in manifest["assets"]:
            for part in asset["processed_parts"]:
                p_file = Path(__file__).resolve().parents[1] / part["processed_filename"]
                self.assertTrue(p_file.is_file(), f"Manifest references missing file {p_file}")

    def test_collision_proxy_excluded_from_perception(self):
        """Verify that group-3 collision proxies never appear in segmentation, depth, or point clouds."""
        scene = WorkshopScene("none", variant="F0_BASE")
        scene.open_container("LEFT_DRAWER")
        scene.open_container("RIGHT_DRAWER")
        scene.open_container("TOOL_CABINET")

        checker = GeometryChecker(scene, width=640, height=480)
        self.assertEqual(checker.allowed_geom_groups, (1,))

        # Inspect segmentation in full reconstruction
        renderer = mujoco.Renderer(scene.model, height=480, width=640)
        scene_opt = checker._build_scene_option()
        renderer.update_scene(scene.data, camera="workshop_camera_front", scene_option=scene_opt)
        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        unique_geoms = set(seg[seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), 0])
        for g_id in unique_geoms:
            g_group = scene.model.geom_group[g_id]
            self.assertNotEqual(g_group, 3, f"Group 3 collision geom {g_id} appeared in perception segmentation!")

        # Verify point clouds for representative objects
        run = checker.run_region_inspection("TOOL_CABINET", rig_config=scene.inspection_rig_config)
        self.assertIn("workshop_long_phillips_driver", run.clouds)
        cloud = run.clouds["workshop_long_phillips_driver"]
        self.assertGreaterEqual(len(cloud.points), MIN_WORKSHOP_OBJECT_FUSED_POINTS)

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
            "fixture_base_col",
            "fixture_clamp_col",
            "frame_horizontal_rail_col",
            "frame_vertical_rail_col",
            "frame_joint_bracket_col",
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
