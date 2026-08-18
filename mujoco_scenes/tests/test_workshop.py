"""Comprehensive test suite for the realistic Workshop (W1) benchmark."""

import json
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
    WORKSHOP_HARDWARE_BIN_CAVITY_VOLUME_M3,
    WORKSHOP_HARDWARE_BIN_HEIGHT_M,
    WORKSHOP_HARDWARE_BIN_LENGTH_M,
    WORKSHOP_HARDWARE_BIN_WIDTH_M,
    WORKSHOP_INSPECTION_RIG_CONFIG,
    WORKSHOP_PARTS_TRAY_CAVITY_VOLUME_M3,
    WORKSHOP_PARTS_TRAY_HEIGHT_M,
    WORKSHOP_PARTS_TRAY_LENGTH_M,
    WORKSHOP_PARTS_TRAY_WIDTH_M,
    WORKSHOP_REGIONS,
    WorkshopScene,
    _load_workshop_variants_config,
    privileged_actual_parts_container_regions,
    privileged_actual_storage_region,
    privileged_actual_work_surface_regions,
    privileged_validate_variant_feasibility,
)


def _get_body_collision_aabb(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact world-space axis-aligned bounding box of all collision geoms for a body and its subtree."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body not found: {body_name}")
    min_pt = np.array([np.inf, np.inf, np.inf])
    max_pt = np.array([-np.inf, -np.inf, -np.inf])
    for gid in range(model.ngeom):
        gbid = model.geom_bodyid[gid]
        curr = gbid
        is_child = False
        while curr >= 0:
            if curr == bid:
                is_child = True
                break
            if curr == 0:
                break
            curr = model.body_parentid[curr]
        if is_child and model.geom_group[gid] == 3:
            gpos = data.geom_xpos[gid]
            gmat = data.geom_xmat[gid].reshape(3, 3)
            gsize = model.geom_size[gid]
            gtype = model.geom_type[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                corners = np.array([
                    [sx * gsize[0], sy * gsize[1], sz * gsize[2]]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
                ])
                world_corners = (gmat @ corners.T).T + gpos
                min_pt = np.minimum(min_pt, world_corners.min(axis=0))
                max_pt = np.maximum(max_pt, world_corners.max(axis=0))
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r, h = gsize[0], gsize[1]
                corners = np.array([
                    [sx * r, sy * r, sz * h]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
                ])
                world_corners = (gmat @ corners.T).T + gpos
                min_pt = np.minimum(min_pt, world_corners.min(axis=0))
                max_pt = np.maximum(max_pt, world_corners.max(axis=0))
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = gsize[0]
                min_pt = np.minimum(min_pt, gpos - r)
                max_pt = np.maximum(max_pt, gpos + r)
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, h = gsize[0], gsize[1]
                min_pt = np.minimum(min_pt, gpos - np.array([r, r, h + r]))
                max_pt = np.maximum(max_pt, gpos + np.array([r, r, h + r]))
    return min_pt, max_pt


def _get_body_geom_aabbs(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact world-space visual (geom_group=1) and collision (geom_group=3) AABBs for a body and its subtree."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body not found: {body_name}")
    vis_min = np.array([np.inf, np.inf, np.inf])
    vis_max = np.array([-np.inf, -np.inf, -np.inf])
    col_min = np.array([np.inf, np.inf, np.inf])
    col_max = np.array([-np.inf, -np.inf, -np.inf])
    for gid in range(model.ngeom):
        gbid = model.geom_bodyid[gid]
        curr = gbid
        is_child = False
        while curr >= 0:
            if curr == bid:
                is_child = True
                break
            if curr == 0:
                break
            curr = model.body_parentid[curr]
        if is_child:
            gpos = data.geom_xpos[gid]
            gmat = data.geom_xmat[gid].reshape(3, 3)
            gtype = model.geom_type[gid]
            ggrp = model.geom_group[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[gid]
                v_start = model.mesh_vertadr[mid]
                v_num = model.mesh_vertnum[mid]
                verts = model.mesh_vert[v_start:v_start+v_num]
                wverts = (gmat @ verts.T).T + gpos
                bmin, bmax = wverts.min(axis=0), wverts.max(axis=0)
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                gsz = model.geom_size[gid]
                corners = np.array([[sx*gsz[0], sy*gsz[1], sz*gsz[2]] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
                wcorners = (gmat @ corners.T).T + gpos
                bmin, bmax = wcorners.min(axis=0), wcorners.max(axis=0)
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r, h = model.geom_size[gid][0], model.geom_size[gid][1]
                corners = np.array([[sx*r, sy*r, sz*h] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
                wcorners = (gmat @ corners.T).T + gpos
                bmin, bmax = wcorners.min(axis=0), wcorners.max(axis=0)
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = model.geom_size[gid][0]
                bmin, bmax = gpos - r, gpos + r
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, h = model.geom_size[gid][0], model.geom_size[gid][1]
                bmin = gpos - np.array([r, r, h + r])
                bmax = gpos + np.array([r, r, h + r])
            else:
                continue
            if ggrp == 1:
                vis_min = np.minimum(vis_min, bmin)
                vis_max = np.maximum(vis_max, bmax)
            else:
                col_min = np.minimum(col_min, bmin)
                col_max = np.maximum(col_max, bmax)
    return vis_min, vis_max, col_min, col_max


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
        self.assertAlmostEqual(cab_f6[0], -0.49, delta=0.05)

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
        # Must stay on cabinet shelf (z >= 0.82)
        self.assertGreaterEqual(pos_c[2], 0.82)

    def test_all_workshop_storage_objects_are_physically_well_placed(self):
        """Verify across all 14 variants that storage objects rest on support, stay contained, and do not overlap."""
        config = _load_workshop_variants_config()
        variants = config.get("variants", {})

        for vname, vspec in variants.items():
            scene = WorkshopScene("none", variant=vname)
            is_swapped = vname == "F6_LAYOUT_SWAPPED"
            cab_x = -0.49 if is_swapped else 0.44

            cab_usable = {
                "min": [cab_x - 0.137, 0.56 - 0.094, 0.688],
                "max": [cab_x + 0.137, 0.56 + 0.084, 0.978],
            }
            left_usable = {"min": [-0.43, 0.16, 0.472], "max": [-0.13, 0.53, 0.60]}
            right_usable = {"min": [0.13, 0.16, 0.472], "max": [0.43, 0.53, 0.60]}
            reg_bounds = {"TOOL_CABINET": cab_usable, "LEFT_DRAWER": left_usable, "RIGHT_DRAWER": right_usable}
            support_z = {"TOOL_CABINET": (0.8260, 0.6880), "LEFT_DRAWER": (0.4720,), "RIGHT_DRAWER": (0.4720,)}

            for reg_id, obj_list in vspec.get("storage_contents", {}).items():
                aabbs = []
                for obj_name in obj_list:
                    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                    self.assertGreaterEqual(bid, 0, f"Body {obj_name} missing in {vname}")

                    min_pt = np.array([np.inf, np.inf, np.inf])
                    max_pt = np.array([-np.inf, -np.inf, -np.inf])
                    for gid in range(scene.model.ngeom):
                        if scene.model.geom_bodyid[gid] == bid and scene.model.geom_group[gid] == 1:
                            gpos = scene.data.geom_xpos[gid]
                            gmat = scene.data.geom_xmat[gid].reshape(3, 3)
                            gtype = scene.model.geom_type[gid]
                            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                                mid = scene.model.geom_dataid[gid]
                                vert_start = scene.model.mesh_vertadr[mid]
                                vert_num = scene.model.mesh_vertnum[mid]
                                verts = scene.model.mesh_vert[vert_start:vert_start + vert_num]
                                world_verts = (gmat @ verts.T).T + gpos
                                min_pt = np.minimum(min_pt, world_verts.min(axis=0))
                                max_pt = np.maximum(max_pt, world_verts.max(axis=0))
                            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                                gsize = scene.model.geom_size[gid]
                                corners = np.array([[sx * gsize[0], sy * gsize[1], sz * gsize[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                                world_corners = (gmat @ corners.T).T + gpos
                                min_pt = np.minimum(min_pt, world_corners.min(axis=0))
                                max_pt = np.maximum(max_pt, world_corners.max(axis=0))

                    aabbs.append((obj_name, min_pt, max_pt))

                    # 1. Containment check
                    ub = reg_bounds[reg_id]
                    for axis, name in enumerate(["X", "Y", "Z"]):
                        self.assertGreaterEqual(min_pt[axis], ub["min"][axis] - 0.005, f"{vname} {reg_id} {obj_name} min {name} < {ub['min'][axis]}")
                        self.assertLessEqual(max_pt[axis], ub["max"][axis] + 0.005, f"{vname} {reg_id} {obj_name} max {name} > {ub['max'][axis]}")

                    # 2. Support contact check (within 5mm)
                    min_gap = min(abs(min_pt[2] - sz) for sz in support_z[reg_id])
                    self.assertLessEqual(min_gap, 0.005, f"{vname} {reg_id} {obj_name} support contact gap {min_gap:.4f} > 5mm")

                # 3. Pairwise non-intersection
                for i, (n1, min1, max1) in enumerate(aabbs):
                    for n2, min2, max2 in aabbs[i + 1:]:
                        overlap_x = max(0.0, min(max1[0], max2[0]) - max(min1[0], min2[0]))
                        overlap_y = max(0.0, min(max1[1], max2[1]) - max(min1[1], min2[1]))
                        overlap_z = max(0.0, min(max1[2], max2[2]) - max(min1[2], min2[2]))
                        vol = overlap_x * overlap_y * overlap_z
                        self.assertEqual(vol, 0.0, f"{vname} {reg_id}: {n1} overlaps {n2}")

    def test_workshop_tabletop_props_rest_on_support_surface(self):
        """Verify that tabletop props rest directly on the wood tabletop (z=0.68m) in visual and collision geometry."""
        for vname in ("F0_BASE", "F2_REGION_ALTERNATIVE"):
            scene = WorkshopScene("none", variant=vname)
            _, w_vmax, _, w_cmax = _get_body_geom_aabbs(scene.model, scene.data, "workbench")
            self.assertAlmostEqual(w_vmax[2], 0.6800, delta=0.003, msg="Workbench visual top is not 0.68m")
            self.assertAlmostEqual(w_cmax[2], 0.6800, delta=0.003, msg="Workbench collision top is not 0.68m")

            tabletop_bodies = [
                "workshop_parts_tray",
                "workshop_hardware_bin",
                "workshop_frame_fixture",
                "tool_cabinet",
            ]
            if vname == "F2_REGION_ALTERNATIVE":
                tabletop_bodies.append("workbench_surface_obstruction")

            for bname in tabletop_bodies:
                vmin, _, cmin, _ = _get_body_geom_aabbs(scene.model, scene.data, bname)
                # Visual bottom must match tabletop height z=0.6800m within 3mm
                self.assertAlmostEqual(
                    vmin[2],
                    0.6800,
                    delta=0.003,
                    msg=f"{vname} {bname} visual bottom ({vmin[2]:.4f}) does not match tabletop height 0.68m",
                )
                # Collision bottom must match tabletop height z=0.6800m within 3mm
                self.assertAlmostEqual(
                    cmin[2],
                    0.6800,
                    delta=0.003,
                    msg=f"{vname} {bname} collision bottom ({cmin[2]:.4f}) does not match tabletop height 0.68m",
                )

    def test_cabinet_door_sweep_does_not_intersect_stored_objects(self):
        """Verify across sampled hinge angles that opening the cabinet door does not intersect stored objects."""
        sampled_angles = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.45]
        for vname in ("F0_BASE", "F5_DECOY_HEAVY", "F6_LAYOUT_SWAPPED"):
            scene = WorkshopScene("none", variant=vname)
            door_jnt = scene.model.joint("tool_cabinet_door_hinge").id
            door_col_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "tool_cabinet_door_col")
            door_col_sz = scene.model.geom_size[door_col_id]
            stored_objs = scene.storage_contents.get("TOOL_CABINET", [])

            for angle in sampled_angles:
                scene.data.qpos[door_jnt] = angle
                mujoco.mj_forward(scene.model, scene.data)

                # Transformed door collision box in world frame
                d_pos = scene.data.geom_xpos[door_col_id]
                d_mat = scene.data.geom_xmat[door_col_id].reshape(3, 3)
                corners = np.array([
                    [sx * door_col_sz[0], sy * door_col_sz[1], sz * door_col_sz[2]]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
                ])
                wcorners = (d_mat @ corners.T).T + d_pos
                door_min = wcorners.min(axis=0)
                door_max = wcorners.max(axis=0)

                for obj_name in stored_objs:
                    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                    obj_min = np.array([np.inf, np.inf, np.inf])
                    obj_max = np.array([-np.inf, -np.inf, -np.inf])
                    for gid in range(scene.model.ngeom):
                        if scene.model.geom_bodyid[gid] == bid:
                            gpos = scene.data.geom_xpos[gid]
                            gmat = scene.data.geom_xmat[gid].reshape(3, 3)
                            gtype = scene.model.geom_type[gid]
                            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                                mid = scene.model.geom_dataid[gid]
                                v_start = scene.model.mesh_vertadr[mid]
                                v_num = scene.model.mesh_vertnum[mid]
                                verts = scene.model.mesh_vert[v_start:v_start + v_num]
                                wverts = (gmat @ verts.T).T + gpos
                                obj_min = np.minimum(obj_min, wverts.min(axis=0))
                                obj_max = np.maximum(obj_max, wverts.max(axis=0))
                            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                                gsz = scene.model.geom_size[gid]
                                cn = np.array([[sx * gsz[0], sy * gsz[1], sz * gsz[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                                wc = (gmat @ cn.T).T + gpos
                                obj_min = np.minimum(obj_min, wc.min(axis=0))
                                obj_max = np.maximum(obj_max, wc.max(axis=0))

                    # 3D AABB overlap volume between door and stored object
                    overlap_x = max(0.0, min(door_max[0], obj_max[0]) - max(door_min[0], obj_min[0]))
                    overlap_y = max(0.0, min(door_max[1], obj_max[1]) - max(door_min[1], obj_min[1]))
                    overlap_z = max(0.0, min(door_max[2], obj_max[2]) - max(door_min[2], obj_min[2]))
                    vol = overlap_x * overlap_y * overlap_z
                    self.assertEqual(
                        vol,
                        0.0,
                        f"{vname} angle={angle:.2f}rad door collision box intersects {obj_name} (overlap: {overlap_x:.4f}x{overlap_y:.4f}x{overlap_z:.4f})",
                    )

    def test_hardware_bin_and_parts_tray_cross_layer_geometry_consistency(self):
        """Verify cross-layer geometry, bounding dimensions, absolute proposal bounds, and center alignment for parts containers."""
        scene = WorkshopScene("none", variant="F0_BASE")
        manifest_path = Path("mujoco_scenes/assets/workshop_realistic/manifest.json")
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # 1. HARDWARE_BIN checks
        bin_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_hardware_bin")
        self.assertGreaterEqual(bin_bid, 0)
        bin_vmin, bin_vmax, bin_cmin, bin_cmax = _get_body_geom_aabbs(scene.model, scene.data, "workshop_hardware_bin")
        c_dims_bin = [WORKSHOP_HARDWARE_BIN_WIDTH_M, WORKSHOP_HARDWARE_BIN_LENGTH_M, WORKSHOP_HARDWARE_BIN_HEIGHT_M]

        # Physical collision and visual bounding extents: X ~= 0.11, Y ~= 0.15, Z ~= 0.08
        np.testing.assert_allclose(bin_cmax - bin_cmin, c_dims_bin, atol=0.003)
        np.testing.assert_allclose(bin_vmax - bin_vmin, c_dims_bin, atol=0.003)

        # Manifest extents
        bin_entry = next((a for a in manifest["assets"] if a["asset_id"] == "workshop_hardware_bin"), None)
        self.assertIsNotNone(bin_entry)
        bin_part = bin_entry["processed_parts"][0]
        np.testing.assert_allclose(bin_part["canonical_dimensions_m"], c_dims_bin, atol=0.003)

        # Production candidate proposal bounds (must match absolute physical AABB)
        proposals = scene.get_candidate_regions()
        bin_prop = next(
            (p for p in proposals if scene.privileged_backend_name_for_region(p["region_instance_id"]) == "HARDWARE_BIN"),
            None,
        )
        self.assertIsNotNone(bin_prop)
        prop_min = np.array(bin_prop["proposal_bounds_m"]["minimum_world_m"])
        prop_max = np.array(bin_prop["proposal_bounds_m"]["maximum_world_m"])
        np.testing.assert_allclose(prop_min, bin_cmin, atol=0.003)
        np.testing.assert_allclose(prop_max, bin_cmax, atol=0.003)
        np.testing.assert_allclose(prop_max - prop_min, c_dims_bin, atol=0.003)

        # Privileged container specification (center must equal physical AABB center)
        specs = scene.privileged_get_parts_container_specs()
        bin_spec = next((s for s in specs if s["region_id"] == "HARDWARE_BIN"), None)
        self.assertIsNotNone(bin_spec)
        self.assertEqual(bin_spec["dimensions_m"], c_dims_bin)
        np.testing.assert_allclose(bin_spec["center_world_m"], (bin_cmin + bin_cmax) / 2, atol=0.003)
        self.assertAlmostEqual(bin_spec["cavity_volume_m3"], WORKSHOP_HARDWARE_BIN_CAVITY_VOLUME_M3, places=6)

        # 2. PARTS_TRAY checks
        tray_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_parts_tray")
        self.assertGreaterEqual(tray_bid, 0)
        tray_vmin, tray_vmax, tray_cmin, tray_cmax = _get_body_geom_aabbs(scene.model, scene.data, "workshop_parts_tray")
        c_dims_tray = [WORKSHOP_PARTS_TRAY_WIDTH_M, WORKSHOP_PARTS_TRAY_LENGTH_M, WORKSHOP_PARTS_TRAY_HEIGHT_M]

        # Physical collision and visual bounding extents: X ~= 0.22, Y ~= 0.14, Z ~= 0.032
        np.testing.assert_allclose(tray_cmax - tray_cmin, c_dims_tray, atol=0.003)
        np.testing.assert_allclose(tray_vmax - tray_vmin, c_dims_tray, atol=0.003)

        # Manifest extents
        tray_entry = next((a for a in manifest["assets"] if a["asset_id"] == "workshop_parts_tray"), None)
        self.assertIsNotNone(tray_entry)
        tray_part = tray_entry["processed_parts"][0]
        np.testing.assert_allclose(tray_part["canonical_dimensions_m"], c_dims_tray, atol=0.003)

        # Production candidate proposal bounds (must match absolute physical AABB)
        tray_prop = next(
            (p for p in proposals if scene.privileged_backend_name_for_region(p["region_instance_id"]) == "PARTS_TRAY"),
            None,
        )
        self.assertIsNotNone(tray_prop)
        t_prop_min = np.array(tray_prop["proposal_bounds_m"]["minimum_world_m"])
        t_prop_max = np.array(tray_prop["proposal_bounds_m"]["maximum_world_m"])
        np.testing.assert_allclose(t_prop_min, tray_cmin, atol=0.003)
        np.testing.assert_allclose(t_prop_max, tray_cmax, atol=0.003)
        np.testing.assert_allclose(t_prop_max - t_prop_min, c_dims_tray, atol=0.003)

        # Privileged container specification (center must equal physical AABB center)
        tray_spec = next((s for s in specs if s["region_id"] == "PARTS_TRAY"), None)
        self.assertIsNotNone(tray_spec)
        self.assertEqual(tray_spec["dimensions_m"], c_dims_tray)
        np.testing.assert_allclose(tray_spec["center_world_m"], (tray_cmin + tray_cmax) / 2, atol=0.003)
        self.assertAlmostEqual(tray_spec["cavity_volume_m3"], WORKSHOP_PARTS_TRAY_CAVITY_VOLUME_M3, places=6)

    def test_f4_object_region_coupling_inequality_and_packing_logic(self):
        """Verify explicit object-region packing inequality for power driver and surface feasibility."""
        drill_area = PRIVILEGED_WORKSHOP_ORACLE_SPECS["workshop_power_driver"]["bounding_area_m2"]
        screw_area = PRIVILEGED_WORKSHOP_ORACLE_SPECS["workshop_medium_phillips_screw"]["bounding_area_m2"]
        req_set_area = (drill_area + screw_area) * 1.2

        narrow_shelf_area = 0.24 * 0.07  # 0.0168 m^2
        tool_cart_area = 0.32 * 0.21     # 0.0672 m^2

        # Explicit inequality test
        self.assertLess(narrow_shelf_area, req_set_area, f"Narrow shelf area ({narrow_shelf_area:.4f}) should not fit set area ({req_set_area:.4f})")
        self.assertLessEqual(req_set_area, tool_cart_area, f"Tool cart area ({tool_cart_area:.4f}) must fit set area ({req_set_area:.4f})")

        # Privileged oracle validation in F4
        scene_f4 = WorkshopScene("none", variant="F4_OBJECT_REGION_COUPLING")
        f4_res = privileged_validate_variant_feasibility(scene_f4)
        self.assertEqual(f4_res["status"], "FEASIBLE")
        self.assertEqual(f4_res["selected_witness"]["work_surface"], "TOOL_CART_TOP")

        # Privileged oracle validation in I5
        scene_i5 = WorkshopScene("none", variant="I5_OBJECT_REGION_PACKING_FAILURE")
        i5_res = privileged_validate_variant_feasibility(scene_i5)
        self.assertEqual(i5_res["status"], "INFEASIBLE")
        self.assertEqual(i5_res["rejection_reason"], "OBJECT_REGION_PACKING_FAILURE")

        # Privileged oracle validation in I6
        scene_i6 = WorkshopScene("none", variant="I6_GLOBAL_CONFLICT")
        i6_res = privileged_validate_variant_feasibility(scene_i6)
        self.assertEqual(i6_res["status"], "INFEASIBLE")
        self.assertEqual(i6_res["rejection_reason"], "GLOBAL_CONFLICT")

    def test_major_static_workshop_components_do_not_intersect(self):
        """Verify no static collision AABB overlaps between tabletop components in F0 and F6."""
        for var_name in ("F0_BASE", "F6_LAYOUT_SWAPPED"):
            scene = WorkshopScene("none", variant=var_name)
            tabletop_bodies = [
                "workshop_frame_fixture",
                "tool_cabinet",
                "workshop_parts_tray",
                "workshop_hardware_bin",
            ]
            aabbs = {b: _get_body_collision_aabb(scene.model, scene.data, b) for b in tabletop_bodies}

            for i, b1 in enumerate(tabletop_bodies):
                for b2 in tabletop_bodies[i + 1 :]:
                    min1, max1 = aabbs[b1]
                    min2, max2 = aabbs[b2]
                    overlap_3d = np.all(max1 > min2) and np.all(max2 > min1)
                    self.assertFalse(
                        overlap_3d,
                        f"Improper collision overlap between {b1} and {b2} in variant {var_name}",
                    )

            # Assert explicit positive clearance between cabinet and fixture
            cab_min, cab_max = aabbs["tool_cabinet"]
            fix_min, fix_max = aabbs["workshop_frame_fixture"]
            if var_name == "F0_BASE":
                clearance = cab_min[0] - fix_max[0]
            else:
                clearance = fix_min[0] - cab_max[0]
            self.assertGreaterEqual(
                clearance,
                0.020,
                f"Cabinet-fixture clearance in {var_name} ({clearance*100:.1f}cm) is below 2.0cm threshold",
            )

    def test_f6_cabinet_door_sweep_clearance(self):
        """Verify F6 cabinet door articulation trajectory does not intersect fixture or other static elements."""
        scene = WorkshopScene("none", variant="F6_LAYOUT_SWAPPED")
        act_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tool_cabinet_door_actuator")
        door_gid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "tool_cabinet_door_col")
        fix_min, fix_max = _get_body_collision_aabb(scene.model, scene.data, "workshop_frame_fixture")

        for ctrl in np.linspace(0.0, 1.45, 20):
            scene.data.ctrl[act_id] = ctrl
            for _ in range(30):
                mujoco.mj_step(scene.model, scene.data)

            gpos = scene.data.geom_xpos[door_gid]
            gmat = scene.data.geom_xmat[door_gid].reshape(3, 3)
            gsize = scene.model.geom_size[door_gid]
            corners = np.array([
                [sx * gsize[0], sy * gsize[1], sz * gsize[2]]
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
            ])
            door_corners = (gmat @ corners.T).T + gpos
            door_min = door_corners.min(axis=0)
            door_max = door_corners.max(axis=0)

            # Fixture intersection check
            overlap_x = max(0.0, min(door_max[0], fix_max[0]) - max(door_min[0], fix_min[0]))
            overlap_y = max(0.0, min(door_max[1], fix_max[1]) - max(door_min[1], fix_min[1]))
            overlap_z = max(0.0, min(door_max[2], fix_max[2]) - max(door_min[2], fix_min[2]))
            self.assertEqual(
                overlap_x * overlap_y * overlap_z,
                0.0,
                f"F6 cabinet door intersects fixture at ctrl={ctrl:.2f}",
            )
            # Positive clearance along X
            self.assertGreaterEqual(
                fix_min[0] - door_max[0],
                0.020,
                f"F6 cabinet door approaches fixture closer than 2.0cm at ctrl={ctrl:.2f}",
            )

    def test_f6_storage_contents_containment_and_pointcloud(self):
        """Verify that F6 cabinet contents stay within relocated cabinet and pass point-cloud smoke test."""
        scene = WorkshopScene("none", variant="F6_LAYOUT_SWAPPED")
        scene.open_container("TOOL_CABINET")

        cab_min, cab_max = _get_body_collision_aabb(scene.model, scene.data, "tool_cabinet")
        declared_cab_objs = scene.storage_contents.get("TOOL_CABINET", [])
        self.assertEqual(len(declared_cab_objs), 2)

        for obj_name in declared_cab_objs:
            b_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            obj_pos = scene.data.xpos[b_id]
            self.assertGreaterEqual(obj_pos[0], cab_min[0] - 0.05)
            self.assertLessEqual(obj_pos[0], cab_max[0] + 0.05)
            self.assertGreaterEqual(obj_pos[1], cab_min[1] - 0.05)
            self.assertLessEqual(obj_pos[1], cab_max[1] + 0.05)
            self.assertGreaterEqual(obj_pos[2], cab_min[2] - 0.05)

        # Point cloud test
        checker = GeometryChecker(scene, width=640, height=480)
        run = checker.run_region_inspection("TOOL_CABINET", rig_config=scene.inspection_rig_config)
        for obj_name in declared_cab_objs:
            self.assertIn(obj_name, run.clouds)
            point_count = len(run.clouds[obj_name].points)
            self.assertGreaterEqual(
                point_count,
                20,
                f"{obj_name} in F6 TOOL_CABINET only has {point_count} points (<20)",
            )


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
