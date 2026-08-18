"""Integrated Workshop (W1) Joint Object and Region Function Benchmark Scene.

Provides authoritative, config-driven variant construction for the 14-variant
workshop benchmark suite, independent dynamic free-body pickable objects,
deterministic storage slot allocation, physical layout profiles, physical
realization of active functional regions, generic production observation identities,
strict privilege boundaries, and scene-level oracle auditing.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from mujoco_scenes.scene_loader import (
    FREE_CAMERA,
    GOOGLE_ACTUATORS,
    GOOGLE_HOME_QPOS,
    ROBOT_GOOGLE,
    ROBOT_NONE,
    _google_robot_dir,
    _inject_google_robot,
    _load_google_binary_assets,
)


ROOT = Path(__file__).resolve().parent
WORKSHOP_BASE = ROOT / "assets" / "workshop_base.xml"
WORKSHOP_ASSETS_DIR = ROOT / "assets" / "workshop_realistic"
WORKSHOP_INSPECTION_RIG_CONFIG = ROOT / "configs" / "workshop_inspection_rigs.yaml"
WORKSHOP_VARIANTS_CONFIG = ROOT / "configs" / "workshop_variants.yaml"
WORKSHOP_ALTERNATIVES_CONFIG = ROOT / "configs" / "workshop_joint_alternatives.yaml"

WORKSHOP_REGIONS = ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")

WORKSHOP_ALL_FUNCTIONAL_WORK_SURFACES = (
    "MAIN_WORKBENCH_ZONE",
    "TOOL_CART_TOP",
    "NARROW_WALL_SHELF",
)
WORKSHOP_FUNCTIONAL_WORK_SURFACES = WORKSHOP_ALL_FUNCTIONAL_WORK_SURFACES

WORKSHOP_ALL_FUNCTIONAL_PARTS_CONTAINERS = (
    "PARTS_TRAY",
    "HARDWARE_BIN",
)
WORKSHOP_FUNCTIONAL_PARTS_CONTAINERS = WORKSHOP_ALL_FUNCTIONAL_PARTS_CONTAINERS

WORKSHOP_CAMERAS = (
    "workshop_camera_left",
    "workshop_camera_right",
    "workshop_camera_top",
    "workshop_camera_front",
    "workshop_camera_close",
)


# ==============================================================================
# CANONICAL TARGET HOLE & WORKPIECE CONSTANTS
# ==============================================================================

WORKSHOP_TARGET_HOLE_DIAMETER_M = 0.007
WORKSHOP_TARGET_HOLE_RADIUS_M = 0.0035
WORKSHOP_TARGET_HOLE_DEPTH_M = 0.030
WORKSHOP_TARGET_RADIAL_CLEARANCE_M = 0.0005
WORKSHOP_TARGET_RECESS_PROFILE = "PH2"


# ==============================================================================
# PRIVILEGED_SIMULATION_ORACLE_ONLY METADATA
# Used strictly for simulator construction, benchmark auditing, and oracle tests.
# MUST NOT be leaked to production perception or task planning.
# ==============================================================================

PRIVILEGED_WORKSHOP_ORACLE_SPECS: dict[str, dict[str, Any]] = {
    "workshop_long_phillips_driver": {
        "kind": "phillips_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.18,
        "tip_profile": WORKSHOP_TARGET_RECESS_PROFILE,
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.026 * 0.035,
        "mass": 0.15,
    },
    "workshop_stubby_phillips_driver": {
        "kind": "phillips_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.020,  # Insufficient reach for recessed frame joint (<0.025m)
        "tip_profile": WORKSHOP_TARGET_RECESS_PROFILE,
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.014 * 0.040,
        "mass": 0.10,
    },
    "workshop_flathead_screwdriver": {
        "kind": "slotted_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.15,
        "tip_profile": "SLOTTED",  # Incompatible with PH2 fastener recess
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.022 * 0.030,
        "mass": 0.14,
    },
    "workshop_power_driver": {
        "kind": "powered_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.12,
        "tip_profile": WORKSHOP_TARGET_RECESS_PROFILE,
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.21 * 0.08,  # Large footprint: 0.0168 m^2
        "mass": 0.60,
    },
    "workshop_medium_phillips_screw": {
        "kind": "medium_screw",
        "functions": ["can_fasten"],
        "length_m": 0.045,
        "head_diameter_m": 0.014,
        "shaft_diameter_m": 0.0055,
        "recess_profile": WORKSHOP_TARGET_RECESS_PROFILE,
        "recess_width_m": 0.0065,
        "required_tool_reach_m": 0.025,
        "bounding_area_m2": 0.045 * 0.014,
        "mass": 0.02,
    },
    "workshop_short_phillips_screw": {
        "kind": "short_screw",
        "functions": ["can_fasten"],
        "length_m": 0.018,  # Inadequate joint engagement depth (<0.030m)
        "head_diameter_m": 0.014,
        "shaft_diameter_m": 0.0055,
        "recess_profile": WORKSHOP_TARGET_RECESS_PROFILE,
        "recess_width_m": 0.0065,
        "required_tool_reach_m": 0.010,
        "bounding_area_m2": 0.018 * 0.014,
        "mass": 0.01,
    },
    "workshop_hex_bolt": {
        "kind": "hex_bolt",
        "functions": ["can_fasten"],
        "length_m": 0.050,
        "head_diameter_m": 0.018,
        "shaft_diameter_m": 0.008,
        "recess_profile": "HEX",  # Incompatible with PH2 driver
        "recess_width_m": 0.010,
        "required_tool_reach_m": 0.025,
        "bounding_area_m2": 0.050 * 0.018,
        "mass": 0.03,
    },
    "workshop_pliers": {
        "kind": "combination_pliers",
        "functions": ["can_grip"],
        "reach_m": 0.08,
        "bounding_area_m2": 0.19 * 0.05,
        "mass": 0.20,
    },
    "workshop_combination_wrench": {
        "kind": "combination_wrench",
        "functions": ["can_turn_hex"],
        "reach_m": 0.10,
        "bounding_area_m2": 0.21 * 0.04,
        "mass": 0.15,
    },
    "workshop_frame_joint": {
        "kind": "fixture_held_frame_joint",
        "functions": ["target_workpiece"],
        "hole_diameter_m": WORKSHOP_TARGET_HOLE_DIAMETER_M,
        "hole_depth_m": WORKSHOP_TARGET_HOLE_DEPTH_M,
        "radial_clearance_m": WORKSHOP_TARGET_RADIAL_CLEARANCE_M,
        "recess_profile": WORKSHOP_TARGET_RECESS_PROFILE,
    },
}

INITIAL_OBJECTS = (
    ("workshop_frame_joint", "fixture_held_frame_joint"),
)

ALL_PICKABLE_OBJECT_NAMES = tuple(
    name
    for name in PRIVILEGED_WORKSHOP_ORACLE_SPECS.keys()
    if name != "workshop_frame_joint"
)


# ==============================================================================
# REUSABLE SIMULATOR OBJECT TEMPLATES
# Generates top-level free bodies with visual mesh, invisible collision proxy, and freejoint.
# ==============================================================================

def _create_object_element(
    object_name: str, pos: tuple[float, float, float], quat: tuple[float, float, float, float]
) -> ET.Element:
    """Generate an independent free-body MuJoCo XML Element with strict visual/collision separation."""
    body = ET.Element(
        "body",
        {
            "name": object_name,
            "pos": f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}",
            "quat": f"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}",
        },
    )
    ET.SubElement(body, "freejoint", {"name": f"{object_name}_free"})

    spec = PRIVILEGED_WORKSHOP_ORACLE_SPECS.get(object_name, {})
    mass = spec.get("mass", 0.10)
    ET.SubElement(body, "inertial", {"pos": "0 0 0.05", "mass": str(mass), "diaginertia": "0.001 0.001 0.001"})

    if object_name == "workshop_long_phillips_driver":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "long_driver_mesh", "material": "screwdrivers_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "class": "collision", "type": "cylinder", "pos": "0 0 0.05", "size": "0.013 0.05"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "class": "collision", "type": "cylinder", "pos": "0 0 0.165", "size": "0.003 0.065"})

    elif object_name == "workshop_stubby_phillips_driver":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "stubby_driver_mesh", "material": "screwdrivers_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "class": "collision", "type": "cylinder", "pos": "0 0 0.035", "size": "0.015 0.035"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "class": "collision", "type": "cylinder", "pos": "0 0 0.090", "size": "0.003 0.020"})

    elif object_name == "workshop_flathead_screwdriver":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "flathead_driver_mesh", "material": "screwdriver_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "class": "collision", "type": "cylinder", "pos": "0 0 0.05", "size": "0.014 0.05"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "class": "collision", "type": "cylinder", "pos": "0 0 0.15", "size": "0.003 0.05"})

    elif object_name == "workshop_power_driver":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "power_driver_mesh", "material": "drill_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_body", "class": "collision", "type": "box", "pos": "0 0 0.08", "size": "0.035 0.08 0.04"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "class": "collision", "type": "capsule", "fromto": "0 0 0.04 0 -0.08 -0.04", "size": "0.020"})

    elif object_name == "workshop_medium_phillips_screw":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "medium_screw_mesh", "material": "screwdrivers_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "class": "collision", "type": "cylinder", "pos": "0 0 0.0195", "size": "0.00275 0.0195"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_head", "class": "collision", "type": "cylinder", "pos": "0 0 0.042", "size": "0.007 0.003"})

    elif object_name == "workshop_short_phillips_screw":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "short_screw_mesh", "material": "screwdrivers_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "class": "collision", "type": "cylinder", "pos": "0 0 0.006", "size": "0.00275 0.006"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_head", "class": "collision", "type": "cylinder", "pos": "0 0 0.015", "size": "0.007 0.003"})

    elif object_name == "workshop_hex_bolt":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "hex_bolt_mesh", "material": "screwdrivers_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "class": "collision", "type": "cylinder", "pos": "0 0 0.021", "size": "0.004 0.021"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_head", "class": "collision", "type": "cylinder", "pos": "0 0 0.046", "size": "0.009 0.004"})

    elif object_name == "workshop_pliers":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "pliers_mesh", "material": "pliers_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "class": "collision", "type": "box", "pos": "0 0 0.095", "size": "0.025 0.008 0.095"})

    elif object_name == "workshop_combination_wrench":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "combination_wrench_mesh", "material": "wrench_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "class": "collision", "type": "box", "pos": "0 0 0.105", "size": "0.015 0.005 0.105"})

    else:
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "class": "collision", "type": "box", "size": "0.05 0.05 0.05"})

    return body


# ==============================================================================
# DETERMINISTIC STORAGE PLACEMENT SLOTS
# ==============================================================================

def _get_storage_slots(
    layout_swapped: bool = False
) -> dict[str, list[tuple[tuple[float, float, float], tuple[float, float, float, float], str]]]:
    """Return deterministic resting (pos, quat, parent_body) slots for each storage container."""
    q_flat_x = (0.7071, 0.0, 0.7071, 0.0)
    q_flat_y = (0.7071, 0.7071, 0.0, 0.0)

    left_drawer_slots = [
        ((-0.22, 0.36, 0.485), q_flat_x, "left_tool_drawer"),
        ((-0.34, 0.32, 0.485), q_flat_x, "left_tool_drawer"),
        ((-0.22, 0.28, 0.485), q_flat_x, "left_tool_drawer"),
        ((-0.34, 0.40, 0.485), q_flat_x, "left_tool_drawer"),
        ((-0.28, 0.34, 0.485), q_flat_y, "left_tool_drawer"),
    ]

    right_drawer_slots = [
        ((0.22, 0.36, 0.485), q_flat_x, "right_tool_drawer"),
        ((0.34, 0.32, 0.485), q_flat_x, "right_tool_drawer"),
        ((0.22, 0.28, 0.485), q_flat_x, "right_tool_drawer"),
        ((0.34, 0.40, 0.485), q_flat_x, "right_tool_drawer"),
        ((0.28, 0.34, 0.485), q_flat_y, "right_tool_drawer"),
    ]

    cab_x = -0.38 if layout_swapped else 0.38
    cab_y = 0.54
    tool_cabinet_slots = [
        ((cab_x - 0.06, cab_y + 0.04, 0.88), q_flat_x, "tool_cabinet"),
        ((cab_x + 0.06, cab_y - 0.03, 0.88), q_flat_x, "tool_cabinet"),
        ((cab_x + 0.06, cab_y + 0.04, 0.88), q_flat_x, "tool_cabinet"),
        ((cab_x - 0.06, cab_y - 0.03, 0.88), q_flat_x, "tool_cabinet"),
        ((cab_x, cab_y, 0.88), q_flat_y, "tool_cabinet"),
    ]

    return {
        "LEFT_DRAWER": left_drawer_slots,
        "RIGHT_DRAWER": right_drawer_slots,
        "TOOL_CABINET": tool_cabinet_slots,
    }


def _load_workshop_variants_config() -> dict[str, Any]:
    with open(WORKSHOP_VARIANTS_CONFIG, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


# ==============================================================================
# AUTHORITATIVE SCENE BUILDER
# ==============================================================================

def build_workshop_xml(
    robot: str = ROBOT_GOOGLE, variant: str = "F0_BASE"
) -> str:
    """Build the complete, variant-faithful MuJoCo XML string."""
    if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
        raise ValueError("Workshop supports robot google or none")

    config = _load_workshop_variants_config()
    variants = config.get("variants", {})
    if variant not in variants:
        raise ValueError(
            f"Unknown workshop variant {variant!r}. Available: {list(variants.keys())}"
        )

    var_spec = variants[variant]
    storage_contents = var_spec.get("storage_contents", {})
    active_surfaces = set(var_spec.get("active_surfaces", []))
    active_containers = set(var_spec.get("active_containers", []))
    is_swapped = variant == "F6_LAYOUT_SWAPPED"

    root = ET.parse(WORKSHOP_BASE).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Malformed workshop_base.xml: missing <worldbody>")

    # 1. Apply layout transforms for F6_LAYOUT_SWAPPED
    if is_swapped:
        for body in worldbody.iter("body"):
            b_name = body.get("name", "")
            if b_name == "tool_cabinet":
                body.set("pos", "-0.38 0.54 0.71")
            elif b_name == "workshop_tool_cart":
                body.set("pos", "-0.95 0.35 0")
            elif b_name == "workshop_parts_tray":
                body.set("pos", "0.42 0.22 0.71")
            elif b_name == "workshop_hardware_bin":
                body.set("pos", "0.42 0.52 0.71")

    # 2. Apply active/inactive surface modifications
    if "MAIN_WORKBENCH_ZONE" not in active_surfaces:
        obs_body = ET.SubElement(
            worldbody,
            "body",
            {"name": "workbench_surface_obstruction", "pos": "-0.02 0.28 0.73"},
        )
        ET.SubElement(
            obs_body,
            "geom",
            {
                "name": "obstruction_case_vis",
                "class": "visual",
                "type": "box",
                "size": "0.16 0.11 0.035",
                "material": "obstruction_case_mat",
            },
        )
        ET.SubElement(
            obs_body,
            "geom",
            {
                "name": "obstruction_case_col",
                "class": "collision",
                "type": "box",
                "size": "0.16 0.11 0.035",
            },
        )

    if "TOOL_CART_TOP" not in active_surfaces:
        for body in list(worldbody):
            if body.get("name") == "workshop_tool_cart":
                worldbody.remove(body)

    # Optional NARROW_WALL_SHELF: present only when explicitly declared in active_surfaces
    if "NARROW_WALL_SHELF" in active_surfaces:
        shelf_pos = "0.70 0.68 1.05" if is_swapped else "-0.70 0.68 1.05"
        shelf_body = ET.SubElement(
            worldbody,
            "body",
            {"name": "workshop_narrow_shelf", "pos": shelf_pos},
        )
        ET.SubElement(
            shelf_body,
            "geom",
            {
                "name": "narrow_shelf_bracket_l",
                "class": "visual",
                "type": "box",
                "pos": "-0.18 -0.08 -0.08",
                "size": "0.015 0.08 0.08",
                "material": "dark_steel",
            },
        )
        ET.SubElement(
            shelf_body,
            "geom",
            {
                "name": "narrow_shelf_bracket_r",
                "class": "visual",
                "type": "box",
                "pos": "0.18 -0.08 -0.08",
                "size": "0.015 0.08 0.08",
                "material": "dark_steel",
            },
        )
        ET.SubElement(
            shelf_body,
            "geom",
            {
                "name": "narrow_shelf_top_vis",
                "class": "visual",
                "type": "box",
                "pos": "0 -0.08 0",
                "size": "0.24 0.07 0.015",
                "material": "shelf_wood",
            },
        )
        ET.SubElement(
            shelf_body,
            "geom",
            {
                "name": "narrow_shelf_top_col",
                "class": "collision",
                "type": "box",
                "pos": "0 -0.08 0",
                "size": "0.24 0.07 0.015",
            },
        )

    # 3. Apply active/inactive parts container modifications
    if "PARTS_TRAY" not in active_containers:
        for body in list(worldbody):
            if body.get("name") == "workshop_parts_tray":
                worldbody.remove(body)

    if "HARDWARE_BIN" not in active_containers:
        for body in list(worldbody):
            if body.get("name") == "workshop_hardware_bin":
                worldbody.remove(body)

    # 4. Instantiate declared storage objects into deterministic slots
    slots = _get_storage_slots(layout_swapped=is_swapped)
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    present_pickable_objects: set[str] = set()

    for region_id, object_list in storage_contents.items():
        region_slots = slots.get(region_id, [])
        for idx, obj_name in enumerate(object_list):
            if idx >= len(region_slots):
                base_slot = region_slots[-1]
                pos = (base_slot[0][0], base_slot[0][1] + 0.08 * (idx - len(region_slots) + 1), base_slot[0][2])
                quat = base_slot[1]
                parent_body = base_slot[2]
            else:
                pos, quat, parent_body = region_slots[idx]

            obj_elem = _create_object_element(obj_name, pos, quat)
            worldbody.append(obj_elem)
            present_pickable_objects.add(obj_name)

            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"storage_weld_{obj_name}",
                    "body1": parent_body,
                    "body2": obj_name,
                    "active": "true",
                    "solref": "0.01 1",
                },
            )

    # 5. Inject Google Robot if requested
    if robot == ROBOT_GOOGLE:
        _inject_google_robot(root, _google_robot_dir())
        for object_name in present_pickable_objects:
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"google:pick_weld_{object_name}",
                    "body1": "google:link_gripper",
                    "body2": object_name,
                    "active": "false",
                    "solref": "0.01 1",
                },
            )

    return ET.tostring(root, encoding="unicode")


# ==============================================================================
# PRIVILEGED PHYSICAL SCENE AUDIT HELPERS
# Evaluates compiled MjModel structure without relying on YAML metadata.
# ==============================================================================

def privileged_actual_storage_region(
    scene: WorkshopScene, object_name: str
) -> str:
    """Privileged scene audit helper checking actual physical MuJoCo body position."""
    body_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
    )
    if body_id < 0:
        return "none"
    x, y, z = scene.data.xpos[body_id]

    is_swapped = scene.variant_name == "F6_LAYOUT_SWAPPED"
    cab_x_min = -0.58 if is_swapped else 0.18
    cab_x_max = -0.18 if is_swapped else 0.58

    if cab_x_min <= x <= cab_x_max and 0.40 <= y <= 0.70 and 0.70 <= z <= 1.20:
        return "TOOL_CABINET"

    if -0.50 <= x <= -0.10 and 0.15 <= y <= 0.60 and 0.35 <= z <= 0.65:
        return "LEFT_DRAWER"

    if 0.10 <= x <= 0.50 and 0.15 <= y <= 0.60 and 0.35 <= z <= 0.65:
        return "RIGHT_DRAWER"

    if -0.80 <= x <= 0.80 and 0.0 <= y <= 0.80 and 0.65 <= z <= 1.20:
        return "workbench"

    return "workspace"


def privileged_actual_work_surface_regions(scene: WorkshopScene) -> list[str]:
    """Privileged physical audit inspecting actual compiled MjModel for available surfaces."""
    available: list[str] = []

    # 1. MAIN_WORKBENCH_ZONE: workbench must exist and must not be obstructed
    has_workbench = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workbench") >= 0
    has_obstruction = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workbench_surface_obstruction") >= 0
    if has_workbench and not has_obstruction:
        available.append("MAIN_WORKBENCH_ZONE")

    # 2. TOOL_CART_TOP: tool cart body must physically exist
    if mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_tool_cart") >= 0:
        available.append("TOOL_CART_TOP")

    # 3. NARROW_WALL_SHELF: narrow shelf body must physically exist
    if mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_narrow_shelf") >= 0:
        available.append("NARROW_WALL_SHELF")

    return available


def privileged_actual_parts_container_regions(scene: WorkshopScene) -> list[str]:
    """Privileged physical audit inspecting actual compiled MjModel for available containers."""
    available: list[str] = []

    if mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_parts_tray") >= 0:
        available.append("PARTS_TRAY")

    if mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_hardware_bin") >= 0:
        available.append("HARDWARE_BIN")

    return available


def privileged_validate_variant_feasibility(
    scene: WorkshopScene,
) -> dict[str, Any]:
    """Privileged oracle feasibility validator analyzing actual physical scene and regions."""
    present_body_names = {
        mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(scene.model.nbody)
    }

    target_hole_diam = WORKSHOP_TARGET_HOLE_DIAMETER_M
    target_joint_depth = WORKSHOP_TARGET_HOLE_DEPTH_M
    radial_clearance = WORKSHOP_TARGET_RADIAL_CLEARANCE_M

    # Derive candidate regions strictly from compiled physical MjModel
    physical_surface_ids = set(privileged_actual_work_surface_regions(scene))
    physical_container_ids = set(privileged_actual_parts_container_regions(scene))

    candidate_surfaces = [
        s for s in scene.privileged_get_work_surface_specs()
        if s["region_id"] in physical_surface_ids
    ]
    candidate_containers = [
        c for c in scene.privileged_get_parts_container_specs()
        if c["region_id"] in physical_container_ids
    ]

    # Find present drivers and fasteners
    present_drivers = []
    present_fasteners = []

    for name in present_body_names:
        if name not in PRIVILEGED_WORKSHOP_ORACLE_SPECS:
            continue
        spec = PRIVILEGED_WORKSHOP_ORACLE_SPECS[name]
        fns = spec.get("functions", [])
        if "can_drive_screw" in fns:
            present_drivers.append((name, spec))
        if "can_fasten" in fns:
            present_fasteners.append((name, spec))

    # Evaluate 4-tuples: (driver, fastener, surface, container)
    valid_witnesses = []

    for d_name, d_spec in present_drivers:
        for f_name, f_spec in present_fasteners:
            f_diam = f_spec.get("shaft_diameter_m", 0.010)
            fits_hole = (f_diam + 2.0 * radial_clearance) <= target_hole_diam
            f_len = f_spec.get("length_m", 0.0)
            reaches_joint = f_len >= target_joint_depth
            d_profile = d_spec.get("tip_profile", "")
            f_profile = f_spec.get("recess_profile", "")
            tip_mates = bool(d_profile and f_profile and d_profile.upper() == f_profile.upper())
            d_reach = d_spec.get("reach_m", 0.0)
            req_reach = f_spec.get("required_tool_reach_m", 0.025)
            driver_reaches = d_reach >= req_reach
            req_set_area = (d_spec.get("bounding_area_m2", 0.01) + f_spec.get("bounding_area_m2", 0.001)) * 1.2

            for surf in candidate_surfaces:
                usable_area = surf.get("usable_area_m2", 0.0)
                fits_surface = usable_area >= req_set_area

                for cont in candidate_containers:
                    fits_container = cont.get("is_open", True) and cont.get("cavity_volume_m3", 0.0) > 0.0

                    if (
                        fits_hole
                        and reaches_joint
                        and tip_mates
                        and driver_reaches
                        and fits_surface
                        and fits_container
                    ):
                        valid_witnesses.append(
                            {
                                "driver": d_name,
                                "fastener": f_name,
                                "work_surface": surf["region_id"],
                                "parts_container": cont["region_id"],
                            }
                        )

    if valid_witnesses:
        return {
            "status": "FEASIBLE",
            "feasible": True,
            "witness_count": len(valid_witnesses),
            "selected_witness": valid_witnesses[0],
            "rejection_reason": None,
        }

    # Determine exact specific failure mode
    rejection_reason = "GLOBAL_CONFLICT"

    if not present_drivers:
        rejection_reason = "NO_VALID_DRIVER"
    elif not any(
        (f_spec.get("shaft_diameter_m", 0.01) + 2.0 * radial_clearance) <= target_hole_diam
        and f_spec.get("length_m", 0.0) >= target_joint_depth
        for _, f_spec in present_fasteners
    ):
        rejection_reason = "NO_VALID_FASTENER"
    elif not candidate_surfaces:
        rejection_reason = "NO_WORK_SURFACE"
    elif not candidate_containers:
        rejection_reason = "NO_PARTS_CONTAINER"
    else:
        all_drivers_short = bool(present_drivers) and all(
            d_spec.get("reach_m", 0.0) < 0.025 for _, d_spec in present_drivers
        )
        if all_drivers_short:
            rejection_reason = "TOOL_GEOMETRY_FAILURE"
        else:
            max_surf_area = max((s.get("usable_area_m2", 0.0) for s in candidate_surfaces), default=0.0)
            mating_pairs = [
                (d_name, d_spec, f_name, f_spec)
                for d_name, d_spec in present_drivers
                for f_name, f_spec in present_fasteners
                if (f_spec.get("shaft_diameter_m", 0.01) + 2.0 * radial_clearance) <= target_hole_diam
                and f_spec.get("length_m", 0.0) >= target_joint_depth
                and d_spec.get("tip_profile", "").upper() == f_spec.get("recess_profile", "").upper()
            ]
            valid_tool_sets = [
                (d_spec.get("bounding_area_m2", 0.01) + f_spec.get("bounding_area_m2", 0.001)) * 1.2
                for _, d_spec, _, f_spec in mating_pairs
                if d_spec.get("reach_m", 0.0) >= f_spec.get("required_tool_reach_m", 0.025)
            ]
            if valid_tool_sets and all(set_area > max_surf_area for set_area in valid_tool_sets):
                rejection_reason = "OBJECT_REGION_PACKING_FAILURE"

    return {
        "status": "INFEASIBLE",
        "feasible": False,
        "witness_count": 0,
        "selected_witness": None,
        "rejection_reason": rejection_reason,
    }


# ==============================================================================
# WORKSHOP SCENE CLASS
# ==============================================================================

@dataclass
class WorkshopObservationState:
    opened_containers: set[str] = field(default_factory=set)
    container_open_state: dict[str, bool] = field(
        default_factory=lambda: {region_id: False for region_id in WORKSHOP_REGIONS}
    )
    joint_repaired: bool = False


class WorkshopScene:
    """Workshop geometry plus bounded closed-region observation API."""

    scene_name = "W1_workshop_joint_alternatives"
    goal = (
        "Repair the loose frame joint using an appropriate tool and fastener. "
        "Arrange the required tool and hardware on a suitable nearby work surface, "
        "and keep loose small parts in a suitable container."
    )
    point_cloud_cameras = WORKSHOP_CAMERAS
    perception_render_geom_groups = (0, 1, 2)
    perception_instance_geom_groups = (1,)
    perception_geom_groups = (1,)
    inspection_rig_config_path = WORKSHOP_INSPECTION_RIG_CONFIG
    initial_observation_region = "workbench"
    default_inspection_order = WORKSHOP_REGIONS
    inspection_interference: dict[str, str] = {}

    def __init__(self, robot: str = ROBOT_GOOGLE, variant: str = "F0_BASE"):
        if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
            raise ValueError("Workshop supports robot google or none")

        self.robot_name = robot
        self.has_robot = robot == ROBOT_GOOGLE
        self.variant_name = variant
        self.state = WorkshopObservationState()

        # Load variant spec
        config = _load_workshop_variants_config()
        self.variant_meta = config["variants"].get(variant, {})
        self.storage_contents = self.variant_meta.get("storage_contents", {})
        self.active_surfaces = self.variant_meta.get(
            "active_surfaces", list(WORKSHOP_ALL_FUNCTIONAL_WORK_SURFACES)
        )
        self.active_containers = self.variant_meta.get(
            "active_containers", list(WORKSHOP_ALL_FUNCTIONAL_PARTS_CONTAINERS)
        )

        # Build persistent deterministic generic instance IDs (object_0001, ...)
        all_possible_objects: list[str] = [name for name, _ in INITIAL_OBJECTS]
        for region_objects in self.storage_contents.values():
            for obj_name in region_objects:
                if obj_name not in all_possible_objects:
                    all_possible_objects.append(obj_name)
        all_possible_objects.sort()

        self._backend_to_instance_id: dict[str, str] = {
            name: f"object_{idx + 1:04d}"
            for idx, name in enumerate(all_possible_objects)
        }
        self._instance_to_backend_id: dict[str, str] = {
            v: k for k, v in self._backend_to_instance_id.items()
        }

        # Deterministic region proposals
        all_regions = list(WORKSHOP_ALL_FUNCTIONAL_WORK_SURFACES) + list(WORKSHOP_ALL_FUNCTIONAL_PARTS_CONTAINERS)
        all_regions.sort()
        self._backend_to_region_id: dict[str, str] = {
            name: f"region_{idx + 1:04d}"
            for idx, name in enumerate(all_regions)
        }
        self._region_to_backend_id: dict[str, str] = {
            v: k for k, v in self._backend_to_region_id.items()
        }

        # Load binary mesh/texture assets for MjModel
        assets = {
            f"workshop_realistic/{path.name}": path.read_bytes()
            for path in WORKSHOP_ASSETS_DIR.iterdir()
            if path.suffix.lower() in {".obj", ".png", ".mtl"}
        }
        if self.has_robot:
            assets.update(_load_google_binary_assets(_google_robot_dir()))

        xml_str = build_workshop_xml(robot=robot, variant=variant)
        self.model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)
        self.data = mujoco.MjData(self.model)
        self.reset()

    def reset(self, settle_steps: int = 300) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.state = WorkshopObservationState()

        if self.has_robot:
            for joint_name, value in GOOGLE_HOME_QPOS.items():
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if joint_id >= 0:
                    self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
            for actuator_name, joint_name, *_rest in GOOGLE_ACTUATORS:
                actuator_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = GOOGLE_HOME_QPOS[joint_name]

        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    # --------------------------------------------------------------------------
    # Production-Safe Observation APIs (Zero Ground-Truth Leaks)
    # --------------------------------------------------------------------------

    def get_observed_instances(self) -> list[dict[str, Any]]:
        """Return currently observed objects with generic IDs and source region."""
        visible = [name for name, _ in INITIAL_OBJECTS]
        for region_id in self.state.opened_containers:
            for obj_name in self.storage_contents.get(region_id, []):
                visible.append(obj_name)
        return [
            {
                "instance_id": self._backend_to_instance_id[name],
                "source_region": self.get_instance_source_region(name),
            }
            for name in visible
        ]

    def get_instance_source_region(self, instance_name: str) -> str | None:
        backend_name = self._instance_to_backend_id.get(instance_name, instance_name)
        if any(name == backend_name for name, _ in INITIAL_OBJECTS):
            return "workbench"
        for region_id, objects in self.storage_contents.items():
            if backend_name in objects:
                return region_id if region_id in self.state.opened_containers else None
        return None

    def inspection_source_region(self, region_id: str) -> str | None:
        return "workbench" if region_id == "INITIAL" else region_id

    def get_region_observation_states(self) -> dict[str, dict]:
        return {
            region_id: {
                "region_id": region_id,
                "open": self.state.container_open_state[region_id],
                "inspected": region_id in self.state.opened_containers,
            }
            for region_id in WORKSHOP_REGIONS
        }

    def get_candidate_regions(self) -> list[dict[str, Any]]:
        """Return neutral candidate spatial region proposals based strictly on physical scene presence."""
        proposals = []

        # 1. MAIN_WORKBENCH_ZONE: physical workbench body exists in the room
        wb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "workbench")
        if wb_id >= 0:
            wb_pos = self.data.xpos[wb_id]
            center = [wb_pos[0] + 0.05, wb_pos[1] - 0.06, wb_pos[2] + 0.68]
            dim = [0.50, 0.32, 0.04]
            proposals.append(
                {
                    "region_instance_id": self._backend_to_region_id["MAIN_WORKBENCH_ZONE"],
                    "proposal_bounds_m": {
                        "minimum_world_m": [center[0] - dim[0] / 2, center[1] - dim[1] / 2, center[2] - dim[2] / 2],
                        "maximum_world_m": [center[0] + dim[0] / 2, center[1] + dim[1] / 2, center[2] + dim[2] / 2],
                    },
                    "observation_source": "workbench",
                }
            )

        # 2. TOOL_CART_TOP: mobile tool cart physically present
        cart_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_tool_cart")
        if cart_id >= 0:
            cart_pos = self.data.xpos[cart_id]
            center = [cart_pos[0], cart_pos[1], cart_pos[2] + 0.80]
            dim = [0.32, 0.21, 0.05]
            proposals.append(
                {
                    "region_instance_id": self._backend_to_region_id["TOOL_CART_TOP"],
                    "proposal_bounds_m": {
                        "minimum_world_m": [center[0] - dim[0] / 2, center[1] - dim[1] / 2, center[2] - dim[2] / 2],
                        "maximum_world_m": [center[0] + dim[0] / 2, center[1] + dim[1] / 2, center[2] + dim[2] / 2],
                    },
                    "observation_source": "cart",
                }
            )

        # 3. NARROW_WALL_SHELF: wall shelf physically present
        shelf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_narrow_shelf")
        if shelf_id >= 0:
            shelf_pos = self.data.xpos[shelf_id]
            center = [shelf_pos[0], shelf_pos[1], shelf_pos[2]]
            dim = [0.24, 0.07, 0.03]
            proposals.append(
                {
                    "region_instance_id": self._backend_to_region_id["NARROW_WALL_SHELF"],
                    "proposal_bounds_m": {
                        "minimum_world_m": [center[0] - dim[0] / 2, center[1] - dim[1] / 2, center[2] - dim[2] / 2],
                        "maximum_world_m": [center[0] + dim[0] / 2, center[1] + dim[1] / 2, center[2] + dim[2] / 2],
                    },
                    "observation_source": "shelf",
                }
            )

        # 4. PARTS_TRAY: parts tray physically present
        tray_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_parts_tray")
        if tray_id >= 0:
            tray_pos = self.data.xpos[tray_id]
            center = [tray_pos[0], tray_pos[1], tray_pos[2]]
            dim = [0.22, 0.14, 0.035]
            proposals.append(
                {
                    "region_instance_id": self._backend_to_region_id["PARTS_TRAY"],
                    "proposal_bounds_m": {
                        "minimum_world_m": [center[0] - dim[0] / 2, center[1] - dim[1] / 2, center[2] - dim[2] / 2],
                        "maximum_world_m": [center[0] + dim[0] / 2, center[1] + dim[1] / 2, center[2] + dim[2] / 2],
                    },
                    "observation_source": "workbench",
                }
            )

        # 5. HARDWARE_BIN: hardware bin physically present
        bin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_hardware_bin")
        if bin_id >= 0:
            bin_pos = self.data.xpos[bin_id]
            center = [bin_pos[0], bin_pos[1], bin_pos[2]]
            dim = [0.15, 0.11, 0.08]
            proposals.append(
                {
                    "region_instance_id": self._backend_to_region_id["HARDWARE_BIN"],
                    "proposal_bounds_m": {
                        "minimum_world_m": [center[0] - dim[0] / 2, center[1] - dim[1] / 2, center[2] - dim[2] / 2],
                        "maximum_world_m": [center[0] + dim[0] / 2, center[1] + dim[1] / 2, center[2] + dim[2] / 2],
                    },
                    "observation_source": "workbench",
                }
            )

        return proposals

    def get_target_workpiece_specification(self) -> dict[str, Any]:
        """Return neutral target workpiece localization."""
        return {
            "target_instance_id": self._backend_to_instance_id.get("workshop_frame_joint", "target_0001"),
            "fixture_center_world_m": [-0.08, 0.30, 0.71],
        }

    # --------------------------------------------------------------------------
    # Privileged Oracle APIs (Explicitly Marked for Evaluation & Benchmark Audit)
    # --------------------------------------------------------------------------

    def privileged_get_visible_backend_instances(self) -> list[tuple[str, str]]:
        """Privileged oracle helper returning (backend_body_name, generic_instance_id)."""
        visible = [name for name, _ in INITIAL_OBJECTS]
        for region_id in self.state.opened_containers:
            for obj_name in self.storage_contents.get(region_id, []):
                visible.append(obj_name)
        return [(name, self._backend_to_instance_id[name]) for name in visible]

    def privileged_backend_name_for_instance(self, instance_id: str) -> str:
        """Privileged helper resolving generic instance ID to backend body name."""
        return self._instance_to_backend_id.get(instance_id, instance_id)

    def privileged_instance_id_for_backend(self, backend_name: str) -> str:
        """Privileged helper resolving backend body name to generic instance ID."""
        return self._backend_to_instance_id.get(backend_name, backend_name)

    def privileged_backend_name_for_region(self, region_instance_id: str) -> str:
        """Privileged helper resolving generic region ID to backend region name."""
        return self._region_to_backend_id.get(region_instance_id, region_instance_id)

    def privileged_region_instance_id_for_backend(self, backend_region: str) -> str:
        """Privileged helper resolving backend region name to generic region ID."""
        return self._backend_to_region_id.get(backend_region, backend_region)

    def privileged_get_storage_contents(self, region_id: str) -> list[str]:
        """Privileged simulation oracle helper returning declared backend storage contents."""
        return list(self.storage_contents.get(region_id, []))

    def privileged_get_work_surface_specs(self) -> list[dict[str, Any]]:
        """Privileged oracle helper returning exact ground-truth work surface properties."""
        all_candidates = [
            {
                "region_id": "MAIN_WORKBENCH_ZONE",
                "center_world_m": [0.05, 0.32, 0.68],
                "dimensions_m": [0.50, 0.32, 0.04],
                "usable_area_m2": 0.50 * 0.32,
            },
            {
                "region_id": "TOOL_CART_TOP",
                "center_world_m": [
                    -0.95 if self.variant_name == "F6_LAYOUT_SWAPPED" else 0.95,
                    0.35,
                    0.80,
                ],
                "dimensions_m": [0.32, 0.21, 0.05],
                "usable_area_m2": 0.32 * 0.21,
            },
            {
                "region_id": "NARROW_WALL_SHELF",
                "center_world_m": [
                    0.70 if self.variant_name == "F6_LAYOUT_SWAPPED" else -0.70,
                    0.68,
                    1.05,
                ],
                "dimensions_m": [0.24, 0.07, 0.03],
                "usable_area_m2": 0.24 * 0.07,
            },
        ]
        return [c for c in all_candidates if c["region_id"] in self.active_surfaces]

    def privileged_get_parts_container_specs(self) -> list[dict[str, Any]]:
        """Privileged oracle helper returning exact ground-truth parts container properties."""
        all_candidates = [
            {
                "region_id": "PARTS_TRAY",
                "center_world_m": [
                    0.42 if self.variant_name == "F6_LAYOUT_SWAPPED" else -0.42,
                    0.22,
                    0.71,
                ],
                "dimensions_m": [0.22, 0.14, 0.035],
                "cavity_volume_m3": 0.20 * 0.12 * 0.026,
                "is_open": True,
            },
            {
                "region_id": "HARDWARE_BIN",
                "center_world_m": [
                    0.42 if self.variant_name == "F6_LAYOUT_SWAPPED" else -0.42,
                    0.52,
                    0.71,
                ],
                "dimensions_m": [0.15, 0.11, 0.08],
                "cavity_volume_m3": 0.14 * 0.10 * 0.07,
                "is_open": True,
            },
        ]
        return [c for c in all_candidates if c["region_id"] in self.active_containers]

    def privileged_get_target_joint_specification(self) -> dict[str, Any]:
        """Privileged oracle helper returning exact workpiece ground truth."""
        return {
            "workpiece_id": "workshop_frame_joint",
            "fixture_center_world_m": [-0.08, 0.30, 0.71],
            "target_hole_diameter_m": WORKSHOP_TARGET_HOLE_DIAMETER_M,
            "target_hole_depth_m": WORKSHOP_TARGET_HOLE_DEPTH_M,
            "target_radial_clearance_m": WORKSHOP_TARGET_RADIAL_CLEARANCE_M,
            "required_driver_function": "can_drive_screw",
            "required_fastener_function": "can_fasten",
            "required_recess_profile": WORKSHOP_TARGET_RECESS_PROFILE,
        }

    def privileged_get_ground_truth_solution(self) -> dict[str, Any] | None:
        """Privileged oracle helper for ground-truth benchmark auditing."""
        return self.variant_meta.get("expected_solution")

    def privileged_get_variant_metadata(self) -> dict[str, Any]:
        """Privileged oracle metadata for test verification."""
        return dict(self.variant_meta)

    # --------------------------------------------------------------------------
    # Scene Dynamics & Articulation
    # --------------------------------------------------------------------------

    @property
    def inspection_rig_config(self) -> dict[str, Any]:
        """Return variant-aware inspection rig configuration."""
        from mujoco_scenes.geometry_checker import load_inspection_rig_config
        raw_config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
        config = copy.deepcopy(raw_config)
        if self.variant_name == "F6_LAYOUT_SWAPPED":
            if "TOOL_CABINET" in config.get("regions", {}):
                cab_reg = config["regions"]["TOOL_CABINET"]
                cab_reg["target_world_m"] = [-0.38, 0.54, 0.88]
                cab_reg["rig_position_world_m"] = [-0.38, -0.70, 1.25]
                cab_reg["inspection_volume"]["minimum_world_m"] = [-0.65, 0.35, 0.68]
                cab_reg["inspection_volume"]["maximum_world_m"] = [-0.15, 0.75, 1.15]
        return config

    def get_task_scene_state(self) -> dict[str, Any]:
        """Return production-safe task observation state (zero privileged metadata)."""
        return {
            "joint_access": {
                "clear": True,
            },
            "containers": {
                region_id: {"open": self.state.container_open_state[region_id]}
                for region_id in WORKSHOP_REGIONS
            },
        }

    def privileged_get_task_scene_state(self) -> dict[str, Any]:
        """Privileged simulation oracle helper returning complete ground-truth scene state."""
        return {
            "variant": self.variant_name,
            "joint_repaired": self.state.joint_repaired,
            "joint_access": {
                "clear": True,
                "covered_by": None,
            },
            "containers": {
                region_id: {"open": self.state.container_open_state[region_id]}
                for region_id in WORKSHOP_REGIONS
            },
        }

    def _container_actuator_id(self, region_id: str) -> int:
        actuator_name = {
            "LEFT_DRAWER": "left_tool_drawer_actuator",
            "RIGHT_DRAWER": "right_tool_drawer_actuator",
            "TOOL_CABINET": "tool_cabinet_door_actuator",
        }.get(region_id)
        if actuator_name is None:
            raise ValueError(f"Unknown workshop region: {region_id}")
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if actuator_id < 0:
            raise RuntimeError(f"Missing actuator: {actuator_name}")
        return actuator_id

    def open_container(self, region_id: str, steps: int = 600) -> dict[str, Any]:
        if region_id not in WORKSHOP_REGIONS:
            raise ValueError(f"Unknown storage container: {region_id}")
        actuator_id = self._container_actuator_id(region_id)
        target = 1.45 if region_id == "TOOL_CABINET" else 0.40
        self.data.ctrl[actuator_id] = target
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        was_new = region_id not in self.state.opened_containers
        self.state.container_open_state[region_id] = True
        self.state.opened_containers.add(region_id)
        return {
            "region_id": region_id,
            "opened": True,
            "newly_opened": was_new,
        }

    def close_container(self, region_id: str, steps: int = 600) -> None:
        actuator_id = self._container_actuator_id(region_id)
        self.data.ctrl[actuator_id] = 0.0
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.state.container_open_state[region_id] = False

    def print_scene_summary(self) -> None:
        print(f"Scene: {self.scene_name}")
        print(f"Variant: {self.variant_name} ({self.variant_meta.get('intended_outcome', 'UNKNOWN')})")
        print(f"Goal:  {self.goal}")
        print(f"Robot: {self.robot_name}")
        print(f"Inspected regions: {sorted(self.state.opened_containers)}")
        print("Storage regions: " + ", ".join(WORKSHOP_REGIONS))

    def render_frame(
        self,
        camera: str = "workshop_camera_front",
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        if camera not in WORKSHOP_CAMERAS:
            raise ValueError(f"Unknown workshop camera: {camera}")
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        scene_option = mujoco.MjvOption()
        if hasattr(self, "perception_render_geom_groups") and self.perception_render_geom_groups is not None:
            scene_option.geomgroup[:] = 0
            for g in self.perception_render_geom_groups:
                if 0 <= g < len(scene_option.geomgroup):
                    scene_option.geomgroup[g] = 1
        elif hasattr(self, "perception_geom_groups") and self.perception_geom_groups is not None:
            scene_option.geomgroup[:] = 0
            for g in self.perception_geom_groups:
                if 0 <= g < len(scene_option.geomgroup):
                    scene_option.geomgroup[g] = 1
        try:
            renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
            return renderer.render().copy()
        finally:
            renderer.close()

    def launch_viewer(self, camera: str = FREE_CAMERA) -> None:
        if camera != FREE_CAMERA and camera not in WORKSHOP_CAMERAS:
            raise ValueError(f"Unknown workshop camera: {camera}")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            if camera == FREE_CAMERA:
                mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
                )
            while viewer.is_running():
                started = time.time()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)


def privileged_audit_object_dimensions(scene: WorkshopScene) -> dict[str, dict[str, Any]]:
    """Privileged audit helper reporting visual mesh dimensions, collision proxy AABB, oracle specs, and manifest metadata."""
    manifest_path = WORKSHOP_ASSETS_DIR / "manifest.json"
    manifest_data = {}
    if manifest_path.is_file():
        try:
            m_json = json.loads(manifest_path.read_text(encoding="utf-8"))
            for asset in m_json.get("assets", []):
                roles = asset.get("roles", [])
                for part in asset.get("processed_parts", []):
                    manifest_data[part["part_id"]] = part
                    for role in roles:
                        if role not in manifest_data:
                            manifest_data[role] = part
        except Exception:
            pass

    report: dict[str, dict[str, Any]] = {}
    for obj_name in ALL_PICKABLE_OBJECT_NAMES:
        b_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if b_id < 0:
            continue

        col_min = np.array([np.inf, np.inf, np.inf])
        col_max = np.array([-np.inf, -np.inf, -np.inf])
        has_col = False
        for g_id in range(scene.model.ngeom):
            if scene.model.geom_bodyid[g_id] == b_id and scene.model.geom_group[g_id] == 3:
                has_col = True
                g_pos = scene.model.geom_pos[g_id]
                g_size = scene.model.geom_size[g_id]
                g_type = scene.model.geom_type[g_id]
                if g_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    half_extent = np.array([g_size[0], g_size[0], g_size[1]])
                elif g_type == mujoco.mjtGeom.mjGEOM_BOX:
                    half_extent = g_size
                elif g_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    half_extent = np.array([g_size[0], g_size[0], g_size[0]])
                else:
                    half_extent = np.zeros(3)
                col_min = np.minimum(col_min, g_pos - half_extent)
                col_max = np.maximum(col_max, g_pos + half_extent)

        collision_extents = (col_max - col_min).tolist() if has_col else None

        visual_extents = None
        for g_id in range(scene.model.ngeom):
            if scene.model.geom_bodyid[g_id] == b_id and scene.model.geom_group[g_id] == 1:
                data_meshid = scene.model.geom_dataid[g_id]
                if data_meshid >= 0:
                    vert_addr = scene.model.mesh_vertadr[data_meshid]
                    vert_num = scene.model.mesh_vertnum[data_meshid]
                    verts = scene.model.mesh_vert[vert_addr : vert_addr + vert_num]
                    visual_extents = (verts.max(axis=0) - verts.min(axis=0)).tolist()

        oracle_spec = PRIVILEGED_WORKSHOP_ORACLE_SPECS.get(obj_name, {})
        manifest_entry = manifest_data.get(obj_name, {})

        report[obj_name] = {
            "visual_mesh_extents_m": visual_extents,
            "collision_proxy_extents_m": collision_extents,
            "oracle_spec": oracle_spec,
            "manifest_canonical_extents_m": manifest_entry.get("canonical_dimensions_m"),
        }

    return report


def list_variants() -> None:
    config = _load_workshop_variants_config()
    variants = config.get("variants", {})
    print(f"Workshop Variants ({len(variants)} total):")
    print(f"{'Variant Name':<34} {'Outcome':<12} Description")
    print("-" * 80)
    for v_name, v_data in variants.items():
        outcome = v_data.get("intended_outcome", "UNKNOWN")
        desc = v_data.get("description", "")
        print(f"{v_name:<34} {outcome:<12} {desc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=(ROBOT_GOOGLE, ROBOT_NONE), default=ROBOT_GOOGLE)
    parser.add_argument("--variant", default="F0_BASE", help="Workshop variant to initialize.")
    parser.add_argument("--list-variants", action="store_true", help="List all available variants.")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--camera", default=FREE_CAMERA)
    parser.add_argument("--open", choices=WORKSHOP_REGIONS + ("ALL",), action="append", default=[])
    parser.add_argument("--render", type=str, default=None, help="Render frame to PNG file.")
    arguments = parser.parse_args()

    if arguments.list_variants:
        list_variants()
        return

    scene = WorkshopScene(robot=arguments.robot, variant=arguments.variant)

    for region_id in arguments.open:
        if region_id == "ALL":
            for reg in WORKSHOP_REGIONS:
                scene.open_container(reg)
        else:
            scene.open_container(region_id)

    scene.print_scene_summary()

    if arguments.render:
        from PIL import Image
        cam = arguments.camera if arguments.camera in WORKSHOP_CAMERAS else "workshop_camera_front"
        img_arr = scene.render_frame(camera=cam)
        Image.fromarray(img_arr).save(arguments.render)
        print(f"Rendered view from {cam} saved to {arguments.render}")

    if arguments.viewer:
        scene.launch_viewer(arguments.camera)


if __name__ == "__main__":
    main()
