"""Integrated Workshop (W1) Joint Object and Region Function Benchmark Scene.

Provides authoritative, config-driven variant construction for the 14-variant
workshop benchmark suite, independent dynamic free-body pickable objects,
deterministic storage slot allocation, physical layout profiles, physical
realization of active functional regions, and privileged scene oracle auditing.
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
    "HIGH_CABINET_TOP",
)
WORKSHOP_FUNCTIONAL_WORK_SURFACES = WORKSHOP_ALL_FUNCTIONAL_WORK_SURFACES

WORKSHOP_ALL_FUNCTIONAL_PARTS_CONTAINERS = (
    "PARTS_TRAY",
    "HARDWARE_BIN",
    "TOOLBOX_COMPARTMENT",
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
# PRIVILEGED_SIMULATION_ORACLE_ONLY METADATA
# Used strictly for simulator construction, benchmark auditing, and oracle tests.
# MUST NOT be leaked to production perception or task planning.
# ==============================================================================

PRIVILEGED_WORKSHOP_ORACLE_SPECS: dict[str, dict[str, Any]] = {
    "workshop_long_phillips_driver": {
        "kind": "phillips_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.18,
        "tip_profile": "PH2",
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.026 * 0.035,
        "mass": 0.15,
    },
    "workshop_stubby_phillips_driver": {
        "kind": "phillips_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.020,  # Insufficient reach for recessed frame joint (<0.025m)
        "tip_profile": "PH2",
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
        "tip_profile": "PH2",
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.21 * 0.08,  # Large footprint: 0.0168 m^2
        "mass": 0.60,
    },
    "workshop_medium_phillips_screw": {
        "kind": "medium_screw",
        "functions": ["can_fasten"],
        "length_m": 0.045,
        "head_diameter_m": 0.016,
        "shaft_diameter_m": 0.0055,
        "recess_profile": "PH2",
        "recess_width_m": 0.0065,
        "required_tool_reach_m": 0.025,
        "bounding_area_m2": 0.045 * 0.016,
        "mass": 0.02,
    },
    "workshop_short_phillips_screw": {
        "kind": "short_screw",
        "functions": ["can_fasten"],
        "length_m": 0.018,  # Inadequate joint engagement depth (<0.030m)
        "head_diameter_m": 0.014,
        "shaft_diameter_m": 0.0055,
        "recess_profile": "PH2",
        "recess_width_m": 0.0065,
        "required_tool_reach_m": 0.010,
        "bounding_area_m2": 0.018 * 0.014,
        "mass": 0.01,
    },
    "workshop_long_phillips_screw": {
        "kind": "long_screw",
        "functions": ["can_fasten"],
        "length_m": 0.085,  # Too long for standard hole
        "head_diameter_m": 0.018,
        "shaft_diameter_m": 0.009,
        "recess_profile": "PH2",
        "recess_width_m": 0.0065,
        "required_tool_reach_m": 0.025,
        "bounding_area_m2": 0.085 * 0.018,
        "mass": 0.03,
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
    "workshop_ratchet_wrench": {
        "kind": "ratchet_wrench",
        "functions": ["can_turn_hex"],
        "reach_m": 0.12,
        "bounding_area_m2": 0.24 * 0.045,
        "mass": 0.22,
    },
    "workshop_wooden_mallet": {
        "kind": "wooden_mallet",
        "functions": ["can_hammer"],
        "reach_m": 0.12,
        "bounding_area_m2": 0.28 * 0.11,
        "mass": 0.35,
    },
    "workshop_joint_seal": {
        "kind": "protective_joint_seal",
        "functions": ["seal_cover"],
        "bounding_area_m2": 0.06 * 0.05,
        "mass": 0.12,
    },
    "workshop_frame_joint": {
        "kind": "fixture_held_frame_joint",
        "functions": ["target_workpiece"],
        "hole_diameter_m": 0.007,
        "hole_depth_m": 0.030,
        "radial_clearance_m": 0.0005,
    },
}

INITIAL_OBJECTS = (
    ("workshop_frame_joint", "fixture_held_frame_joint"),
    ("workshop_joint_seal", "protective_joint_seal"),
)

ALL_PICKABLE_OBJECT_NAMES = tuple(
    name
    for name in PRIVILEGED_WORKSHOP_ORACLE_SPECS.keys()
    if name not in ("workshop_frame_joint", "workshop_joint_seal")
)


# ==============================================================================
# REUSABLE SIMULATOR OBJECT TEMPLATES
# Generates top-level free bodies with visual mesh, collision proxy, and freejoint.
# ==============================================================================

def _create_object_element(
    object_name: str, pos: tuple[float, float, float], quat: tuple[float, float, float, float]
) -> ET.Element:
    """Generate an independent free-body MuJoCo XML Element for any workshop object."""
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
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "long_driver_mesh", "material": "screwdrivers_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "type": "cylinder", "pos": "0 0 0.07", "size": "0.020 0.07", "material": "bench_steel"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "type": "cylinder", "pos": "0 0 0.18", "size": "0.004 0.07", "material": "polished_steel"})

    elif object_name == "workshop_stubby_phillips_driver":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "stubby_driver_mesh", "material": "screwdrivers_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "type": "cylinder", "pos": "0 0 0.04", "size": "0.022 0.04", "material": "bench_steel"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "type": "cylinder", "pos": "0 0 0.10", "size": "0.005 0.03", "material": "polished_steel"})

    elif object_name == "workshop_flathead_screwdriver":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "flathead_driver_mesh", "material": "screwdriver_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "type": "cylinder", "pos": "0 0 0.06", "size": "0.018 0.06", "material": "bench_steel"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "type": "cylinder", "pos": "0 0 0.15", "size": "0.004 0.06", "material": "polished_steel"})

    elif object_name == "workshop_power_driver":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "power_driver_mesh", "material": "drill_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_body", "type": "box", "pos": "0 0 0.08", "size": "0.035 0.08 0.04", "material": "bench_steel"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "type": "capsule", "fromto": "0 0 0.04 0 -0.08 -0.04", "size": "0.020", "material": "dark_steel"})

    elif object_name == "workshop_medium_phillips_screw":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "medium_screw_mesh", "material": "screwdrivers_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "type": "cylinder", "pos": "0 0 0.02", "size": "0.008 0.02", "material": "polished_steel"})

    elif object_name == "workshop_short_phillips_screw":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "short_screw_mesh", "material": "screwdrivers_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "type": "cylinder", "pos": "0 0 0.01", "size": "0.008 0.01", "material": "polished_steel"})

    elif object_name == "workshop_long_phillips_screw":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "long_screw_mesh", "material": "screwdrivers_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "type": "cylinder", "pos": "0 0 0.04", "size": "0.009 0.04", "material": "polished_steel"})

    elif object_name == "workshop_hex_bolt":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "hex_bolt_mesh", "material": "screwdrivers_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "type": "cylinder", "pos": "0 0 0.02", "size": "0.010 0.02", "material": "polished_steel"})

    elif object_name == "workshop_pliers":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "pliers_mesh", "material": "pliers_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "type": "box", "pos": "0 0 0.08", "size": "0.025 0.010 0.08", "material": "polished_steel"})

    elif object_name == "workshop_combination_wrench":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "combination_wrench_mesh", "material": "wrench_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "type": "box", "pos": "0 0 0.10", "size": "0.015 0.006 0.10", "material": "polished_steel"})

    elif object_name == "workshop_ratchet_wrench":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "ratchet_wrench_mesh", "material": "ratchet_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "type": "box", "pos": "0 0 0.11", "size": "0.018 0.008 0.11", "material": "polished_steel"})

    elif object_name == "workshop_wooden_mallet":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "type": "mesh", "mesh": "wooden_mallet_mesh", "material": "mallet_visual_mat", "contype": "0", "conaffinity": "0", "group": "1"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_head", "type": "box", "pos": "0 0 0.22", "size": "0.055 0.025 0.035", "material": "bench_wood_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "type": "cylinder", "pos": "0 0 0.10", "size": "0.014 0.10", "material": "bench_wood_mat"})

    else:
        # Generic fallback
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "type": "box", "size": "0.05 0.05 0.05", "material": "bench_steel"})

    return body


# ==============================================================================
# DETERMINISTIC STORAGE PLACEMENT SLOTS
# ==============================================================================

def _get_storage_slots(
    layout_swapped: bool = False
) -> dict[str, list[tuple[tuple[float, float, float], tuple[float, float, float, float], str]]]:
    """Return deterministic (pos, quat, parent_body) slots for each storage container."""
    # Quats: flat lying orientation for long tools: [0.7071, 0, 0.7071, 0] or [0.7071, 0.7071, 0, 0]
    q_flat_x = (0.7071, 0.0, 0.7071, 0.0)
    q_flat_y = (0.7071, 0.7071, 0.0, 0.0)
    q_up = (1.0, 0.0, 0.0, 0.0)

    left_drawer_slots = [
        ((-0.40, 0.28, 0.485), q_flat_x, "left_tool_drawer"),
        ((-0.24, 0.28, 0.485), q_flat_x, "left_tool_drawer"),
        ((-0.40, 0.44, 0.485), q_flat_x, "left_tool_drawer"),
        ((-0.24, 0.44, 0.485), q_flat_x, "left_tool_drawer"),
        ((-0.32, 0.36, 0.485), q_flat_y, "left_tool_drawer"),
    ]

    right_drawer_slots = [
        ((0.24, 0.28, 0.485), q_flat_x, "right_tool_drawer"),
        ((0.40, 0.28, 0.485), q_flat_x, "right_tool_drawer"),
        ((0.24, 0.44, 0.485), q_flat_x, "right_tool_drawer"),
        ((0.40, 0.44, 0.485), q_flat_x, "right_tool_drawer"),
        ((0.32, 0.36, 0.485), q_flat_y, "right_tool_drawer"),
    ]

    cab_x_offset = -0.40 if layout_swapped else 0.0
    tool_cabinet_slots = [
        ((cab_x_offset - 0.08, 0.55, 0.91), q_flat_x, "tool_cabinet"),
        ((cab_x_offset + 0.00, 0.55, 0.91), q_flat_x, "tool_cabinet"),
        ((cab_x_offset + 0.08, 0.55, 0.91), q_flat_x, "tool_cabinet"),
        ((cab_x_offset - 0.08, 0.55, 0.75), q_flat_x, "tool_cabinet"),
        ((cab_x_offset + 0.08, 0.55, 0.75), q_flat_x, "tool_cabinet"),
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
                body.set("pos", "-0.40 0.58 0.71")
            elif b_name == "workshop_tool_cart":
                body.set("pos", "-0.92 0.32 0")
            elif b_name == "workshop_narrow_shelf":
                body.set("pos", "0.70 0.68 1.05")
            elif b_name == "workshop_parts_tray":
                body.set("pos", "0.44 0.22 0.71")
            elif b_name == "workshop_hardware_bin":
                body.set("pos", "-0.44 0.22 0.71")

    # 2. Apply active/inactive surface modifications
    # In F2_REGION_ALTERNATIVE or when MAIN_WORKBENCH_ZONE is inactive: place a physical obstruction
    if "MAIN_WORKBENCH_ZONE" not in active_surfaces:
        obs_body = ET.SubElement(
            worldbody,
            "body",
            {"name": "workbench_surface_obstruction", "pos": "0.0 0.38 0.73"},
        )
        ET.SubElement(
            obs_body,
            "geom",
            {
                "name": "obstruction_crate",
                "type": "box",
                "size": "0.26 0.16 0.04",
                "material": "fixture_steel",
            },
        )
        ET.SubElement(
            obs_body,
            "geom",
            {
                "name": "obstruction_warning_tape",
                "type": "box",
                "pos": "0 0 0.042",
                "size": "0.25 0.15 0.002",
                "material": "tab_yellow",
            },
        )

    # If TOOL_CART_TOP is inactive (e.g. I2, I5, I6), remove/obstruct tool cart
    if "TOOL_CART_TOP" not in active_surfaces:
        for body in list(worldbody):
            if body.get("name") in ("workshop_tool_cart", "workshop_toolbox_compartment"):
                worldbody.remove(body)

    # If NARROW_WALL_SHELF is inactive (e.g. I2), remove narrow shelf
    if "NARROW_WALL_SHELF" not in active_surfaces:
        for body in list(worldbody):
            if body.get("name") == "workshop_narrow_shelf":
                worldbody.remove(body)

    # 3. Apply active/inactive parts container modifications
    if "PARTS_TRAY" not in active_containers:
        for body in list(worldbody):
            if body.get("name") == "workshop_parts_tray":
                worldbody.remove(body)

    if "HARDWARE_BIN" not in active_containers:
        for body in list(worldbody):
            if body.get("name") == "workshop_hardware_bin":
                worldbody.remove(body)

    if "TOOLBOX_COMPARTMENT" not in active_containers:
        for body in list(worldbody):
            if body.get("name") == "workshop_toolbox_compartment":
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
                # Extra objects placed with safe spacing
                base_slot = region_slots[-1]
                pos = (base_slot[0][0], base_slot[0][1] + 0.08 * (idx - len(region_slots) + 1), base_slot[0][2])
                quat = base_slot[1]
                parent_body = base_slot[2]
            else:
                pos, quat, parent_body = region_slots[idx]

            # Create top-level free body
            obj_elem = _create_object_element(obj_name, pos, quat)
            worldbody.append(obj_elem)
            present_pickable_objects.add(obj_name)

            # Add storage weld constraint for deterministic container articulation
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
# PRIVILEGED AUDIT HELPERS
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
    cab_x_min = -0.65 if is_swapped else -0.25
    cab_x_max = -0.15 if is_swapped else 0.25

    # Check TOOL_CABINET bounds
    if cab_x_min <= x <= cab_x_max and 0.40 <= y <= 0.75 and 0.70 <= z <= 1.20:
        return "TOOL_CABINET"

    # Check LEFT_DRAWER bounds
    if -0.55 <= x <= -0.10 and 0.15 <= y <= 0.65 and 0.35 <= z <= 0.65:
        return "LEFT_DRAWER"

    # Check RIGHT_DRAWER bounds
    if 0.10 <= x <= 0.55 and 0.15 <= y <= 0.65 and 0.35 <= z <= 0.65:
        return "RIGHT_DRAWER"

    # Check Workbench general workspace
    if -0.80 <= x <= 0.80 and 0.0 <= y <= 0.80 and 0.65 <= z <= 1.20:
        return "workbench"

    return "workspace"


def privileged_validate_variant_feasibility(
    scene: WorkshopScene,
) -> dict[str, Any]:
    """Privileged oracle feasibility validator analyzing the actual physical scene."""
    present_body_names = {
        mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(scene.model.nbody)
    }

    target_hole_diam = 0.007
    target_joint_depth = 0.030
    radial_clearance = 0.0005

    candidate_surfaces = scene.get_candidate_work_surfaces()
    candidate_containers = scene.get_candidate_parts_containers()

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
            # 1. Fastener fits target hole
            f_diam = f_spec.get("shaft_diameter_m", 0.010)
            fits_hole = (f_diam + 2.0 * radial_clearance) <= target_hole_diam

            # 2. Fastener reaches joint depth
            f_len = f_spec.get("length_m", 0.0)
            reaches_joint = f_len >= target_joint_depth

            # 3. Driver tip matches fastener recess
            d_profile = d_spec.get("tip_profile", "")
            f_profile = f_spec.get("recess_profile", "")
            tip_mates = bool(d_profile and f_profile and d_profile.upper() == f_profile.upper())

            # 4. Driver reach >= required reach
            d_reach = d_spec.get("reach_m", 0.0)
            req_reach = f_spec.get("required_tool_reach_m", 0.025)
            driver_reaches = d_reach >= req_reach

            # 5. Fastener + Driver set fits work surface
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
        # Check if ALL present drivers have insufficient reach (< 0.025m)
        all_drivers_short = bool(present_drivers) and all(
            d_spec.get("reach_m", 0.0) < 0.025 for _, d_spec in present_drivers
        )
        if all_drivers_short:
            rejection_reason = "TOOL_GEOMETRY_FAILURE"
        else:
            # Check packing failure
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
    joint_seal_location: str = "FRAME_JOINT"


class WorkshopScene:
    """Workshop geometry plus bounded closed-region observation API."""

    scene_name = "W1_workshop_joint_alternatives"
    goal = (
        "Repair the loose frame joint using an appropriate tool and fastener. "
        "Arrange the required tool and hardware on a suitable nearby work surface, "
        "and keep loose small parts in a suitable container."
    )
    point_cloud_cameras = WORKSHOP_CAMERAS
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

    def get_visible_object_instances(self) -> list[tuple[str, str]]:
        """Return currently visible objects based on opened container state."""
        visible = list(INITIAL_OBJECTS)
        for region_id in self.state.opened_containers:
            for obj_name in self.storage_contents.get(region_id, []):
                kind = PRIVILEGED_WORKSHOP_ORACLE_SPECS.get(obj_name, {}).get("kind", "tool")
                visible.append((obj_name, kind))
        return visible

    def get_instance_source_region(self, instance_name: str) -> str | None:
        if any(name == instance_name for name, _kind in INITIAL_OBJECTS):
            return "workbench"
        for region_id, objects in self.storage_contents.items():
            if instance_name in objects:
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

    @property
    def inspection_rig_config(self) -> dict[str, Any]:
        """Return variant-aware inspection rig configuration."""
        from mujoco_scenes.geometry_checker import load_inspection_rig_config
        raw_config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
        config = copy.deepcopy(raw_config)
        if self.variant_name == "F6_LAYOUT_SWAPPED":
            # Adjust TOOL_CABINET inspection rig coordinates for swapped cabinet position
            if "TOOL_CABINET" in config.get("regions", {}):
                cab_reg = config["regions"]["TOOL_CABINET"]
                cab_reg["target_world_m"] = [-0.40, 0.55, 0.88]
                cab_reg["rig_position_world_m"] = [-0.40, -0.75, 1.25]
                cab_reg["inspection_volume"]["minimum_world_m"] = [-0.70, 0.38, 0.68]
                cab_reg["inspection_volume"]["maximum_world_m"] = [-0.10, 0.78, 1.18]
        return config

    def get_candidate_work_surfaces(self) -> list[dict[str, Any]]:
        """Return candidate work surface proposals filtered by active variant configuration."""
        all_candidates = [
            {
                "region_id": "MAIN_WORKBENCH_ZONE",
                "center_world_m": [0.0, 0.38, 0.68],
                "dimensions_m": [0.60, 0.33, 0.05],
                "usable_area_m2": 0.60 * 0.33,
            },
            {
                "region_id": "TOOL_CART_TOP",
                "center_world_m": [
                    -0.92 if self.variant_name == "F6_LAYOUT_SWAPPED" else 0.92,
                    0.32,
                    0.80,
                ],
                "dimensions_m": [0.32, 0.21, 0.05],
                "usable_area_m2": 0.32 * 0.21,
            },
            {
                "region_id": "NARROW_WALL_SHELF",
                "center_world_m": [
                    0.70 if self.variant_name == "F6_LAYOUT_SWAPPED" else -0.70,
                    0.60,
                    1.05,
                ],
                "dimensions_m": [0.24, 0.07, 0.03],
                "usable_area_m2": 0.24 * 0.07,
            },
            {
                "region_id": "HIGH_CABINET_TOP",
                "center_world_m": [
                    -0.40 if self.variant_name == "F6_LAYOUT_SWAPPED" else 0.0,
                    0.58,
                    1.10,
                ],
                "dimensions_m": [0.22, 0.14, 0.03],
                "usable_area_m2": 0.22 * 0.14,
            },
        ]
        return [c for c in all_candidates if c["region_id"] in self.active_surfaces]

    def get_candidate_parts_containers(self) -> list[dict[str, Any]]:
        """Return candidate parts container proposals filtered by active variant configuration."""
        all_candidates = [
            {
                "region_id": "PARTS_TRAY",
                "center_world_m": [
                    0.44 if self.variant_name == "F6_LAYOUT_SWAPPED" else -0.44,
                    0.22,
                    0.71,
                ],
                "dimensions_m": [0.32, 0.22, 0.045],
                "cavity_volume_m3": 0.30 * 0.20 * 0.035,
                "is_open": True,
            },
            {
                "region_id": "HARDWARE_BIN",
                "center_world_m": [
                    -0.44 if self.variant_name == "F6_LAYOUT_SWAPPED" else 0.44,
                    0.22,
                    0.71,
                ],
                "dimensions_m": [0.16, 0.12, 0.08],
                "cavity_volume_m3": 0.14 * 0.10 * 0.07,
                "is_open": True,
            },
            {
                "region_id": "TOOLBOX_COMPARTMENT",
                "center_world_m": [
                    -0.92 if self.variant_name == "F6_LAYOUT_SWAPPED" else 0.92,
                    0.32,
                    0.85,
                ],
                "dimensions_m": [0.38, 0.18, 0.14],
                "cavity_volume_m3": 0.36 * 0.16 * 0.12,
                "is_open": True,
            },
        ]
        return [c for c in all_candidates if c["region_id"] in self.active_containers]

    def get_target_joint_specification(self) -> dict[str, Any]:
        """Return target workpiece specification."""
        return {
            "workpiece_id": "workshop_frame_joint",
            "fixture_center_world_m": [-0.02, 0.32, 0.71],
            "target_hole_diameter_m": 0.007,
            "target_hole_depth_m": 0.030,
            "required_driver_function": "can_drive_screw",
            "required_fastener_function": "can_fasten",
            "required_recess_profile": "PH2",
        }

    def get_task_scene_state(self) -> dict[str, Any]:
        return {
            "variant": self.variant_name,
            "joint_repaired": self.state.joint_repaired,
            "joint_access": {
                "clear": self.state.joint_seal_location != "FRAME_JOINT",
                "covered_by": (
                    "workshop_joint_seal"
                    if self.state.joint_seal_location == "FRAME_JOINT"
                    else None
                ),
            },
            "joint_seal_location": self.state.joint_seal_location,
            "containers": {
                region_id: {"open": self.state.container_open_state[region_id]}
                for region_id in WORKSHOP_REGIONS
            },
        }

    def move_joint_seal_to_tray(self, steps: int = 300) -> None:
        """Debug helper to move the seal cover to the parts tray."""
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "workshop_joint_seal_free",
        )
        if joint_id >= 0:
            tray_x = 0.44 if self.variant_name == "F6_LAYOUT_SWAPPED" else -0.44
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr : qpos_adr + 3] = (tray_x, 0.22, 0.73)
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            dof_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
        self.state.joint_seal_location = "PARTS_TRAY"

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

    def open_container(self, region_id: str, steps: int = 300) -> list[str]:
        actuator_id = self._container_actuator_id(region_id)
        target = 1.45 if region_id == "TOOL_CABINET" else 0.40
        self.data.ctrl[actuator_id] = target
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.state.container_open_state[region_id] = True
        was_new = region_id not in self.state.opened_containers
        self.state.opened_containers.add(region_id)
        return list(self.storage_contents.get(region_id, [])) if was_new else []

    def close_container(self, region_id: str, steps: int = 600) -> None:
        actuator_id = self._container_actuator_id(region_id)
        self.data.ctrl[actuator_id] = 0.0
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.state.container_open_state[region_id] = False

    def privileged_get_ground_truth_solution(self) -> dict[str, Any] | None:
        """Privileged oracle helper for ground-truth benchmark auditing."""
        return self.variant_meta.get("expected_solution")

    def privileged_get_variant_metadata(self) -> dict[str, Any]:
        """Privileged oracle metadata for test verification."""
        return dict(self.variant_meta)

    def print_scene_summary(self) -> None:
        print(f"Scene: {self.scene_name}")
        print(f"Variant: {self.variant_name} ({self.variant_meta.get('intended_outcome', 'UNKNOWN')})")
        print(f"Goal:  {self.goal}")
        print(f"Robot: {self.robot_name}")
        print(f"Inspected regions: {sorted(self.state.opened_containers)}")
        print("Storage regions: " + ", ".join(WORKSHOP_REGIONS))
        print("Candidate work surfaces: " + ", ".join([s["region_id"] for s in self.get_candidate_work_surfaces()]))
        print("Candidate parts containers: " + ", ".join([c["region_id"] for c in self.get_candidate_parts_containers()]))

    def render_frame(
        self,
        camera: str = "workshop_camera_front",
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        if camera not in WORKSHOP_CAMERAS:
            raise ValueError(f"Unknown workshop camera: {camera}")
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data, camera=camera)
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
    parser.add_argument("--remove-seal", action="store_true")
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

    if arguments.remove_seal:
        scene.move_joint_seal_to_tray()

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
