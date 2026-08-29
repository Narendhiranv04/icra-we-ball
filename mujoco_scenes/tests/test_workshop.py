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
            scene = WorkshopScene(robot=robot, variant="F0_MANUAL_FIRST_ONE_REGION")
            self.assertEqual(scene.has_robot, robot == "google")
            for camera_name in WORKSHOP_CAMERAS:
                cam_id = mujoco.mj_name2id(
                    scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                )
                self.assertGreaterEqual(cam_id, 0, f"Missing camera: {camera_name}")


    def test_all_10_variants_instantiate_cleanly(self):
        config = _load_workshop_variants_config()
        variants = config.get("variants", {})
        self.assertEqual(len(variants), 10, "Expected exactly 10 benchmark variants (F0-F7, I0-I1)")

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

    def test_workshop_has_three_canonical_views_for_each_observation_stage(self):
        config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
        self.assertEqual(
            config["inspection_sequence"],
            ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
        )
        self.assertEqual(
            set(config["camera_slots"]), {"ISO_LEFT", "ISO_RIGHT", "DETAIL"}
        )
        for region in config["regions"].values():
            self.assertEqual(set(region["cameras"]), set(config["camera_slots"]))
            transforms = {
                (*camera["position_offset_m"], *camera["look_at_offset_m"])
                for camera in region["cameras"].values()
            }
            self.assertEqual(len(transforms), 3)



class WorkshopPrivilegeBoundaryTests(unittest.TestCase):
    pass
