"""Workshop fixed-pair position/presence benchmark scene.

Provides the redesigned 10-variant scene, articulated storage, four fixed
movable-object identities, production-safe observations, oracle auditing, and
an optional interactive robot Actions panel.
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
WORKSHOP_FRAME_FIXTURE_CENTER = (-0.34, 0.34, 0.68)

# The drawer entries deliberately mirror KitchenScene's D1/D2 articulation:
# a 0.25 m prismatic pull-out with the same open threshold.  OPEN is a single
# generic action; the argument selects the drawer or cabinet mechanism.
WORKSHOP_CONTAINER_JOINTS = {
    "LEFT_DRAWER": {
        "joint": "left_tool_drawer_slide",
        "actuator": "left_tool_drawer_actuator",
        "open_val": 0.25,
    },
    "RIGHT_DRAWER": {
        "joint": "right_tool_drawer_slide",
        "actuator": "right_tool_drawer_actuator",
        "open_val": 0.25,
    },
    "TOOL_CABINET": {
        "joint": "tool_cabinet_door_hinge",
        "actuator": "tool_cabinet_door_actuator",
        "open_val": 1.45,
    },
}

# Frozen Workshop-only spawn selected from the base-standoff audit.  The
# shared Kitchen/Living-Room default in scene_loader remains unchanged.
WORKSHOP_GOOGLE_BASE_POSE = {
    "pos": "0 -0.75 0.06205",
    "quat": "0.7071068 0 0 0.7071068",
}

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

WORKSHOP_HARDWARE_BIN_WIDTH_M = 0.11
WORKSHOP_HARDWARE_BIN_LENGTH_M = 0.15
WORKSHOP_HARDWARE_BIN_HEIGHT_M = 0.08
WORKSHOP_HARDWARE_BIN_INNER_WIDTH_M = 0.10
WORKSHOP_HARDWARE_BIN_INNER_LENGTH_M = 0.14
WORKSHOP_HARDWARE_BIN_USABLE_HEIGHT_M = 0.030
WORKSHOP_HARDWARE_BIN_CAVITY_VOLUME_M3 = (
    WORKSHOP_HARDWARE_BIN_INNER_WIDTH_M
    * WORKSHOP_HARDWARE_BIN_INNER_LENGTH_M
    * WORKSHOP_HARDWARE_BIN_USABLE_HEIGHT_M
)

WORKSHOP_PARTS_TRAY_WIDTH_M = 0.22
WORKSHOP_PARTS_TRAY_LENGTH_M = 0.14
WORKSHOP_PARTS_TRAY_HEIGHT_M = 0.032
WORKSHOP_PARTS_TRAY_INNER_WIDTH_M = 0.20
WORKSHOP_PARTS_TRAY_INNER_LENGTH_M = 0.12
WORKSHOP_PARTS_TRAY_USABLE_HEIGHT_M = 0.026
WORKSHOP_PARTS_TRAY_CAVITY_VOLUME_M3 = (
    WORKSHOP_PARTS_TRAY_INNER_WIDTH_M
    * WORKSHOP_PARTS_TRAY_INNER_LENGTH_M
    * WORKSHOP_PARTS_TRAY_USABLE_HEIGHT_M
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
        "bounding_area_m2": 0.163054 * 0.165,  # Compact 12V side-rest planar footprint: ~0.026904 m^2
        # Compact powered screwdriver; light enough for the single-arm
        # platform to carry without the grasp constraint sagging.
        "mass": 0.25,
    },
    "workshop_wooden_hammer": {
        "kind": "wooden_hammer",
        "functions": ["can_strike"],
        "reach_m": 0.16,
        "bounding_area_m2": 0.22 * 0.09,
        "mass": 0.35,
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

ALL_PICKABLE_OBJECT_NAMES = (
    "workshop_long_phillips_driver",
    "workshop_power_driver",
    "workshop_medium_phillips_screw",
    "workshop_wooden_hammer",
)


# ==============================================================================
# REUSABLE SIMULATOR OBJECT TEMPLATES
# Generates top-level free bodies with visual mesh, invisible collision proxy, and freejoint.
# ==============================================================================

def _add_profile_threaded_fastener_visual(
    body: ET.Element, object_name: str, length_m: float, head_radius_m: float
) -> None:
    """Add an opt-in hidden threaded silhouette for detector visual profiles."""
    shaft_radius = 0.0025
    head_half_height = 0.0025
    usable_length = length_m - 2 * head_half_height
    common = {"class": "visual", "group": "5", "rgba": "0 0 0 0"}
    ET.SubElement(body, "geom", {
        **common, "name": f"{object_name}_profile_shaft", "type": "cylinder",
        "pos": f"0 0 {0.5 * usable_length:.4f}",
        "size": f"{shaft_radius:.4f} {0.5 * usable_length:.4f}",
    })
    for arm_name, size_x, size_y in (
        ("head_arm_x", head_radius_m, 0.0015),
    ):
        ET.SubElement(body, "geom", {
            **common, "name": f"{object_name}_profile_{arm_name}", "type": "box",
            "pos": f"0 0 {length_m - head_half_height:.4f}",
            "size": f"{size_x:.4f} {size_y:.4f} {head_half_height:.4f}",
        })
    for index in range(1, 9):
        z = usable_length * index / 10.0
        ET.SubElement(body, "geom", {
            **common, "name": f"{object_name}_profile_thread_{index:02d}",
            "type": "cylinder", "pos": f"0 0 {z:.4f}",
            "size": f"{shaft_radius + 0.0008:.4f} 0.0005",
        })


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
        ET.SubElement(body, "site", {"name": f"{object_name}_handle_site", "pos": "0 0 0.045", "size": "0.004", "rgba": "0 0 0 0"})
        ET.SubElement(body, "site", {"name": f"{object_name}_tip_site", "pos": "0 0 0.230", "size": "0.003", "rgba": "0.2 0.8 1 0.5"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "long_driver_mesh", "material": "screwdrivers_visual_mat"})
        ET.SubElement(body, "geom", {
            "name": f"{object_name}_profile_handle", "class": "visual",
            "type": "cylinder", "pos": "0 0 0.040", "size": "0.013 0.040",
            "rgba": "0 0 0 0", "group": "5",
        })
        ET.SubElement(body, "geom", {
            "name": f"{object_name}_profile_shaft", "class": "visual",
            "type": "cylinder", "pos": "0 0 0.125", "size": "0.003 0.045",
            "rgba": "0 0 0 0", "group": "5",
        })
        ET.SubElement(body, "geom", {
            "name": f"{object_name}_profile_phillips_tip", "class": "visual",
            "type": "box", "pos": "0 0 0.172", "size": "0.004 0.002 0.004",
            "rgba": "0 0 0 0", "group": "5",
        })
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
        # Drill_01's visible chuck/bit points along local -X (the old +Z site
        # described an empty point beside the mesh and made a horizontal drill
        # falsely appear "vertical" numerically).
        ET.SubElement(body, "site", {"name": f"{object_name}_handle_site", "pos": "0.045 0 0.055", "size": "0.006", "rgba": "0 0 0 0"})
        ET.SubElement(body, "site", {"name": f"{object_name}_tip_site", "pos": "-0.0815 0 0.142", "size": "0.004", "rgba": "0.2 0.8 1 0.5"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "power_driver_mesh", "material": "drill_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_body", "class": "collision", "type": "box", "pos": "0 0 0.142", "size": "0.082 0.024 0.023"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "class": "collision", "type": "box", "pos": "0.045 0 0.060", "size": "0.026 0.020 0.060"})

    elif object_name == "workshop_wooden_hammer":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "wooden_hammer_mesh", "material": "wooden_hammer_visual_mat"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_handle", "class": "collision", "type": "cylinder", "pos": "0 0 0.095", "size": "0.011 0.095"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_head", "class": "collision", "type": "box", "pos": "0 0 0.205", "size": "0.045 0.022 0.025"})

    elif object_name == "workshop_medium_phillips_screw":
        # The imported mesh and collision model place the broad Phillips head
        # at the body origin and the pointed end along local +Z.  Keep the
        # semantic sites consistent with what is visibly rendered so a 180
        # degree tip-down placement leaves the broad head above the joint.
        ET.SubElement(body, "site", {"name": f"{object_name}_head_site", "pos": "0 0 0", "size": "0.004", "rgba": "0.2 0.8 1 0.5"})
        ET.SubElement(body, "site", {"name": f"{object_name}_tip_site", "pos": "0 0 0.045", "size": "0.002", "rgba": "0.1 1 0.2 0.5"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "medium_screw_mesh", "material": "screwdrivers_visual_mat"})
        # A 43 mm rendered robust extent stays within the physical 45 mm
        # collision envelope while avoiding a one-pixel excess at the 45 mm
        # target-compatibility boundary.
        _add_profile_threaded_fastener_visual(body, object_name, 0.043, 0.007)
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_shaft", "class": "collision", "type": "cylinder", "pos": "0 0 0.0195", "size": "0.00275 0.0195"})
        ET.SubElement(body, "geom", {"name": f"{object_name}_col_head", "class": "collision", "type": "cylinder", "pos": "0 0 0.042", "size": "0.007 0.003"})

    elif object_name == "workshop_short_phillips_screw":
        ET.SubElement(body, "geom", {"name": f"{object_name}_vis", "class": "visual", "type": "mesh", "mesh": "short_screw_mesh", "material": "screwdrivers_visual_mat"})
        _add_profile_threaded_fastener_visual(body, object_name, 0.018, 0.007)
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
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "class": "collision", "type": "box", "pos": "0 0 0.0067", "size": "0.018 0.105 0.0067"})

    else:
        ET.SubElement(body, "geom", {"name": f"{object_name}_col", "class": "collision", "type": "box", "size": "0.05 0.05 0.05"})

    return body


# ==============================================================================
# DETERMINISTIC STORAGE PLACEMENT SLOTS & NATURAL RESTING POSES
# ==============================================================================

def _get_object_storage_pose(
    obj_name: str, region_id: str, slot_idx: int, layout_swapped: bool = False
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], str]:
    """Return deterministic resting (pos, quat, parent_body) tailored for object physical geometry."""
    q_along_x = (0.7071, 0.0, 0.7071, 0.0)      # local Z -> world +X
    q_along_y = (0.7071, 0.7071, 0.0, 0.0)      # local Z -> world +Y
    q_flat = (1.0, 0.0, 0.0, 0.0)
    q_pliers_x = (0.5, 0.5, 0.5, 0.5)          # local Z -> world +X, local X -> world +Y, local Y -> world +Z
    q_wrench_x = (0.7071, 0.0, 0.0, 0.7071)    # local Y -> world +X
    q_drill_side = (0.7071, -0.7071, 0.0, 0.0) # lies flat on side
    # Drawer-only drill pose: local Z (the tool's long axis) runs across the
    # tray rather than into its shallow front-to-back depth.  This is what
    # lets the copied kitchen drawer close fully around the physical asset.
    q_drill_drawer_x = q_pliers_x

    if region_id == "TOOL_CABINET":
        cab_x = -0.75 if layout_swapped else 0.44
        cab_y = 0.56
        # Lower shelf top after the cabinet accessibility redesign. The extra
        # roof clearance admits the horizontal gripper/wrist around upright
        # tools while the powered driver still fits on the lower floor.
        shelf_z = 0.7660
        floor_z = 0.6880
        cab_slots_xy = [
            (cab_x, cab_y - 0.11),
            (cab_x, cab_y - 0.03),
            (cab_x, cab_y + 0.03),
            (cab_x - 0.04, cab_y - 0.03),
            (cab_x + 0.04, cab_y - 0.03),
        ]
        base_x, base_y = cab_slots_xy[min(slot_idx, len(cab_slots_xy) - 1)]

        if obj_name == "workshop_long_phillips_driver":
            # Stand the driver on the interior shelf, well behind the door
            # plane.  The former door-parented pose made it appear to float on
            # the cabinet face as the door opened.
            return (
                cab_x + (0.12 if slot_idx > 0 else -0.10),
                cab_y - 0.025,
                shelf_z + 0.001,
            ), q_flat, "tool_cabinet"
        elif obj_name == "workshop_medium_phillips_screw":
            # Stand the screw on the shelf. Cabinet retrieval grips its shaft
            # horizontally below the head so the wrist clears the roof.
            return (
                cab_x - 0.10, cab_y - 0.025, shelf_z + 0.001
            ), q_flat, "tool_cabinet"
        elif obj_name == "workshop_short_phillips_screw":
            return (base_x - 0.009, base_y, shelf_z + 0.007), q_along_x, "tool_cabinet"
        elif obj_name == "workshop_stubby_phillips_driver":
            return (base_x - 0.055, base_y, shelf_z + 0.015), q_along_x, "tool_cabinet"
        elif obj_name == "workshop_hex_bolt":
            return (base_x - 0.025, base_y, shelf_z + 0.009), q_along_x, "tool_cabinet"
        elif obj_name == "workshop_power_driver":
            if slot_idx == 0:
                # A selected/sole power driver occupies the accessible centre
                # of the upper shelf for a straight horizontal retrieval.
                return (cab_x, cab_y - 0.03, shelf_z + 0.024), q_drill_side, "tool_cabinet"
            # When it shares the cabinet with the target screw it is an
            # unselected alternative; park it separately on the lower floor
            # so it cannot obstruct the screw's gripper corridor.
            return (cab_x + 0.12, cab_y - 0.03, floor_z + 0.024), q_drill_side, "tool_cabinet"
        elif obj_name == "workshop_wooden_hammer":
            return (cab_x, cab_y - 0.025, shelf_z + 0.001), q_flat, "tool_cabinet"
        elif obj_name == "workshop_pliers":
            return (base_x - 0.095, base_y, shelf_z + 0.010), q_pliers_x, "tool_cabinet"
        elif obj_name == "workshop_combination_wrench":
            return (base_x, base_y, shelf_z), q_wrench_x, "tool_cabinet"
        else:
            return (base_x, base_y, shelf_z + 0.020), q_along_x, "tool_cabinet"

    elif region_id == "LEFT_DRAWER":
        # Kitchen-equivalent tray base top: frame z=0.524, tray base top=.472.
        floor_z = 0.4720
        drawer_slots_xy = [
            (-0.34, 0.07),
            (-0.22, 0.24),
            (-0.22, 0.07),
            (-0.34, 0.24),
            (-0.28, 0.15),
        ]
        base_x, base_y = drawer_slots_xy[min(slot_idx, len(drawer_slots_xy) - 1)]

        if obj_name == "workshop_flathead_screwdriver":
            return (base_x, base_y + 0.110, floor_z + 0.014), q_along_y, "left_tool_drawer"
        elif obj_name == "workshop_long_phillips_driver":
            # The opened kitchen-style tray exposes this front slot directly
            # while retaining the driver's original LEFT_DRAWER membership.
            # The driver's 200 mm collision extent points along +X in this
            # resting orientation, so its origin must remain left of centre
            # to stay completely inside the copied kitchen tray when closed.
            return (-0.36, 0.210, floor_z + 0.013), q_along_x, "left_tool_drawer"
        elif obj_name == "workshop_stubby_phillips_driver":
            return (base_x, base_y + 0.055, floor_z + 0.015), q_along_y, "left_tool_drawer"
        elif obj_name == "workshop_medium_phillips_screw":
            return (-0.34, 0.080, floor_z + 0.007), q_along_x, "left_tool_drawer"
        elif obj_name == "workshop_short_phillips_screw":
            return (base_x, base_y + 0.009, floor_z + 0.007), q_along_y, "left_tool_drawer"
        elif obj_name == "workshop_hex_bolt":
            return (base_x, base_y + 0.025, floor_z + 0.009), q_along_y, "left_tool_drawer"
        elif obj_name == "workshop_power_driver":
            # The powered driver lies across the tray, so the kitchen-size
            # drawer contains its full length while closed.  Keeping it out
            # of the front-back axis also leaves OPEN's approach corridor
            # clear for the robot.
            return (-0.35, 0.150, floor_z + 0.024), q_drill_drawer_x, "left_tool_drawer"
        elif obj_name == "workshop_wooden_hammer":
            # The rendered hammer mesh spans 0.28 m along +X, exceeding its
            # smaller collision proxy.  Keep the complete visible asset
            # between the tray walls, with clearance on both ends.
            return (-0.415, 0.210, floor_z + 0.012), q_along_x, "left_tool_drawer"
        elif obj_name == "workshop_pliers":
            return (base_x, base_y + 0.095, floor_z + 0.010), q_along_y, "left_tool_drawer"
        elif obj_name == "workshop_combination_wrench":
            return (base_x, base_y, floor_z), q_flat, "left_tool_drawer"
        else:
            return (base_x, base_y, floor_z + 0.020), q_along_y, "left_tool_drawer"

    elif region_id == "RIGHT_DRAWER":
        floor_z = 0.4720
        drawer_slots_xy = [
            (0.22, 0.07),
            (0.34, 0.24),
            (0.34, 0.07),
            (0.22, 0.24),
            (0.28, 0.15),
        ]
        base_x, base_y = drawer_slots_xy[min(slot_idx, len(drawer_slots_xy) - 1)]

        if obj_name == "workshop_stubby_phillips_driver":
            return (base_x, base_y + 0.055, floor_z + 0.015), q_along_y, "right_tool_drawer"
        elif obj_name == "workshop_flathead_screwdriver":
            return (base_x, base_y + 0.110, floor_z + 0.014), q_along_y, "right_tool_drawer"
        elif obj_name == "workshop_long_phillips_driver":
            return (0.15, 0.210, floor_z + 0.013), q_along_x, "right_tool_drawer"
        elif obj_name == "workshop_medium_phillips_screw":
            return (0.36, 0.080, floor_z + 0.007), q_along_x, "right_tool_drawer"
        elif obj_name == "workshop_short_phillips_screw":
            return (base_x, base_y + 0.009, floor_z + 0.007), q_along_y, "right_tool_drawer"
        elif obj_name == "workshop_hex_bolt":
            return (base_x, base_y + 0.025, floor_z + 0.009), q_along_y, "right_tool_drawer"
        elif obj_name == "workshop_power_driver":
            return (0.15, 0.150, floor_z + 0.024), q_drill_drawer_x, "right_tool_drawer"
        elif obj_name == "workshop_wooden_hammer":
            return (0.22, 0.210, floor_z + 0.012), q_along_x, "right_tool_drawer"
        elif obj_name == "workshop_pliers":
            return (base_x, base_y + 0.095, floor_z + 0.010), q_along_y, "right_tool_drawer"
        elif obj_name == "workshop_combination_wrench":
            return (base_x, base_y, floor_z), q_flat, "right_tool_drawer"
        else:
            return (base_x, base_y, floor_z + 0.020), q_along_y, "right_tool_drawer"

    return (0.0, 0.0, 0.0), q_flat, "world"


def _get_storage_slots(
    layout_swapped: bool = False
) -> dict[str, list[tuple[tuple[float, float, float], tuple[float, float, float, float], str]]]:
    """Return deterministic resting (pos, quat, parent_body) slots for each storage container."""
    q_flat_x = (0.7071, 0.0, 0.7071, 0.0)
    q_flat_y = (0.7071, 0.7071, 0.0, 0.0)

    # Kitchen-equivalent left tray: floor top Z=.472 m, inner X [-.446, -.114],
    # inner Y [.025, .281] while closed.
    left_drawer_slots = [
        ((-0.34, 0.24, 0.486), q_flat_y, "left_tool_drawer"),
        ((-0.22, 0.119, 0.479), q_flat_y, "left_tool_drawer"),
        ((-0.22, 0.24, 0.486), q_flat_y, "left_tool_drawer"),
        ((-0.34, 0.08, 0.479), q_flat_y, "left_tool_drawer"),
        ((-0.28, 0.15, 0.485), q_flat_y, "left_tool_drawer"),
    ]

    # Kitchen-equivalent right tray: floor top Z=.472 m, inner X [.114, .446],
    # inner Y [.025, .281] while closed.
    right_drawer_slots = [
        ((0.22, 0.185, 0.487), q_flat_y, "right_tool_drawer"),
        ((0.34, 0.155, 0.481), q_flat_y, "right_tool_drawer"),
        ((0.34, 0.24, 0.486), q_flat_y, "right_tool_drawer"),
        ((0.22, 0.08, 0.479), q_flat_y, "right_tool_drawer"),
        ((0.28, 0.15, 0.485), q_flat_y, "right_tool_drawer"),
    ]

    # Tool Cabinet: shelf top Z = 0.8260m, inner X in [cab_x-0.137, cab_x+0.137], inner Y in [cab_y-0.094, cab_y+0.084]
    cab_x = -0.75 if layout_swapped else 0.44
    cab_y = 0.56
    tool_cabinet_slots = [
        ((cab_x - 0.115, cab_y + 0.03, 0.839), q_flat_x, "tool_cabinet"),
        ((cab_x - 0.0225, cab_y - 0.03, 0.833), q_flat_x, "tool_cabinet"),
        ((cab_x - 0.105, cab_y + 0.02, 0.861), q_flat_x, "tool_cabinet"),
        ((cab_x - 0.095, cab_y - 0.03, 0.841), q_flat_x, "tool_cabinet"),
        ((cab_x + 0.05, cab_y - 0.03, 0.833), q_flat_x, "tool_cabinet"),
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
    robot: str = ROBOT_GOOGLE, variant: str = "F0_MANUAL_FIRST_ONE_REGION"
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
    is_swapped = False

    root = ET.parse(WORKSHOP_BASE).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Malformed workshop_base.xml: missing <worldbody>")

    # 1. Apply layout transforms for F6_LAYOUT_SWAPPED
    if is_swapped:
        for body in worldbody.iter("body"):
            b_name = body.get("name", "")
            if b_name == "tool_cabinet":
                body.set("pos", "-0.75 0.56 0.68")
            elif b_name == "workshop_tool_cart":
                body.set("pos", "-1.28 0.40 0")
            elif b_name == "workshop_parts_tray":
                body.set("pos", "0.42 0.22 0.68")
            elif b_name == "workshop_hardware_bin":
                body.set("pos", "0.44 0.52 0.68")

    # 2. Apply active/inactive surface modifications
    if "MAIN_WORKBENCH_ZONE" not in active_surfaces:
        obs_body = ET.SubElement(
            worldbody,
            "body",
            {"name": "workbench_surface_obstruction", "pos": "0.0 0.26 0.715"},
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

    # 4. Instantiate declared storage objects into deterministic natural resting poses
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    present_pickable_objects: set[str] = set()

    for region_id, object_list in storage_contents.items():
        for idx, obj_name in enumerate(object_list):
            pos, quat, parent_body = _get_object_storage_pose(
                obj_name, region_id, idx, layout_swapped=is_swapped
            )

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
        _inject_google_robot(
            root, _google_robot_dir(), base_pose=WORKSHOP_GOOGLE_BASE_POSE
        )
        for region_id, moving_body in (
            ("LEFT_DRAWER", "left_tool_drawer"),
            ("RIGHT_DRAWER", "right_tool_drawer"),
            ("TOOL_CABINET", "tool_cabinet_door"),
        ):
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"workshop_handle_grasp_weld_{region_id}",
                    "body1": "google:link_gripper",
                    "body2": moving_body,
                    "active": "false",
                    "solref": "0.02 1",
                },
            )
        for object_name in present_pickable_objects:
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"google:pick_weld_{object_name}",
                    "body1": "google:link_gripper",
                    "body2": object_name,
                    "active": "false",
                    "solref": (
                        "0.002 1"
                        if object_name == "workshop_power_driver"
                        else "0.01 1"
                    ),
                },
            )
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"workshop_alignment_weld_{object_name}",
                    "body1": "workshop_frame_joint",
                    "body2": object_name,
                    "active": "false",
                    # The joint guide must dominate the compliant hand grasp:
                    # small fasteners and the driver stay concentrically
                    # vertical while the wrist supplies the visible motion.
                    "solref": "0.005 1",
                },
            )
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"workshop_staging_weld_{object_name}",
                    "body1": "workbench",
                    "body2": object_name,
                    "active": "false",
                    "solref": "0.08 1",
                },
            )
            if object_name == "workshop_medium_phillips_screw":
                ET.SubElement(
                    equality,
                    "weld",
                    {
                        "name": "workshop_installed_fastener_weld",
                        "body1": "workshop_frame_joint",
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

    cab_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "tool_cabinet")
    if cab_bid >= 0:
        cab_pos = scene.data.xpos[cab_bid]
        if (
            cab_pos[0] - 0.20 <= x <= cab_pos[0] + 0.20
            and cab_pos[1] - 0.16 <= y <= cab_pos[1] + 0.16
            and cab_pos[2] <= z <= cab_pos[2] + 0.35
        ):
            return "TOOL_CABINET"

    if -0.50 <= x <= -0.10 and -0.20 <= y <= 0.65 and 0.35 <= z <= 0.65:
        return "LEFT_DRAWER"

    if 0.10 <= x <= 0.50 and -0.20 <= y <= 0.65 and 0.35 <= z <= 0.65:
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

    # Evaluate only the two variable functional roles.  The workbench repair
    # hole is a fixed task target, not a VLM-selected alternative region.
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
            if fits_hole and reaches_joint and tip_mates and driver_reaches:
                valid_witnesses.append(
                    {
                        "driver": d_name,
                        "fastener": f_name,
                        "insertion_target": "MAIN_WORKBENCH_ZONE",
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
        rejection_reason = "NO_COMPATIBLE_DRIVER"
    elif not any(
        (f_spec.get("shaft_diameter_m", 0.01) + 2.0 * radial_clearance) <= target_hole_diam
        and f_spec.get("length_m", 0.0) >= target_joint_depth
        for _, f_spec in present_fasteners
    ):
        rejection_reason = "NO_COMPATIBLE_SCREW"
    else:
        all_drivers_short = bool(present_drivers) and all(
            d_spec.get("reach_m", 0.0) < 0.025 for _, d_spec in present_drivers
        )
        if all_drivers_short:
            rejection_reason = "TOOL_GEOMETRY_FAILURE"
        else:
            rejection_reason = "INCOMPATIBLE_DRIVER_SCREW_INTERFACE"

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

    scene_name = "W2_workshop_fixed_pair_position_presence"
    goal = (
        "Find the compatible screw and first compatible driver encountered; "
        "insert the screw tip-down into the workbench hole and drive it fully."
    )
    point_cloud_cameras = WORKSHOP_CAMERAS
    perception_render_geom_groups = (0, 1, 2)
    perception_instance_geom_groups = (1,)
    perception_geom_groups = (1,)
    inspection_rig_config_path = WORKSHOP_INSPECTION_RIG_CONFIG
    initial_observation_region = "workbench"
    default_inspection_order = WORKSHOP_REGIONS
    inspection_interference: dict[str, str] = {}

    def __init__(self, robot: str = ROBOT_GOOGLE, variant: str = "F0_MANUAL_FIRST_ONE_REGION"):
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

        # A Workshop start is always the physical closed state.  Resetting an
        # action-panel scene after an OPEN must not inherit actuator commands
        # or joint velocity from the preceding run.
        for mechanism in WORKSHOP_CONTAINER_JOINTS.values():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, mechanism["joint"]
            )
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mechanism["actuator"]
            )
            if joint_id < 0 or actuator_id < 0:
                raise RuntimeError(
                    "Workshop storage reset is missing its required "
                    f"joint or actuator: {mechanism}"
                )
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
            self.data.ctrl[actuator_id] = 0.0

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
            center = [wb_pos[0], wb_pos[1] - 0.12, wb_pos[2] + 0.68]
            dim = [0.50, 0.28, 0.04]
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
            dim = [
                WORKSHOP_PARTS_TRAY_WIDTH_M,
                WORKSHOP_PARTS_TRAY_LENGTH_M,
                WORKSHOP_PARTS_TRAY_HEIGHT_M,
            ]
            center = [tray_pos[0], tray_pos[1], tray_pos[2] + dim[2] / 2]
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
            dim = [
                WORKSHOP_HARDWARE_BIN_WIDTH_M,
                WORKSHOP_HARDWARE_BIN_LENGTH_M,
                WORKSHOP_HARDWARE_BIN_HEIGHT_M,
            ]
            center = [bin_pos[0], bin_pos[1], bin_pos[2] + dim[2] / 2]
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
            "fixture_center_world_m": list(WORKSHOP_FRAME_FIXTURE_CENTER),
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
                "center_world_m": [0.0, 0.26, 0.68],
                "dimensions_m": [0.50, 0.28, 0.04],
                "usable_area_m2": 0.50 * 0.28,
            },
            {
                "region_id": "TOOL_CART_TOP",
                "center_world_m": [
                    -1.28 if self.variant_name == "F6_LAYOUT_SWAPPED" else 1.28,
                    0.40,
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
                    0.68 + WORKSHOP_PARTS_TRAY_HEIGHT_M / 2,
                ],
                "dimensions_m": [
                    WORKSHOP_PARTS_TRAY_WIDTH_M,
                    WORKSHOP_PARTS_TRAY_LENGTH_M,
                    WORKSHOP_PARTS_TRAY_HEIGHT_M,
                ],
                "cavity_volume_m3": WORKSHOP_PARTS_TRAY_CAVITY_VOLUME_M3,
                "is_open": True,
            },
            {
                "region_id": "HARDWARE_BIN",
                "center_world_m": [
                    0.44 if self.variant_name == "F6_LAYOUT_SWAPPED" else -0.44,
                    0.52,
                    0.68 + WORKSHOP_HARDWARE_BIN_HEIGHT_M / 2,
                ],
                "dimensions_m": [
                    WORKSHOP_HARDWARE_BIN_WIDTH_M,
                    WORKSHOP_HARDWARE_BIN_LENGTH_M,
                    WORKSHOP_HARDWARE_BIN_HEIGHT_M,
                ],
                "cavity_volume_m3": WORKSHOP_HARDWARE_BIN_CAVITY_VOLUME_M3,
                "is_open": True,
            },
        ]
        return [c for c in all_candidates if c["region_id"] in self.active_containers]

    def privileged_get_target_joint_specification(self) -> dict[str, Any]:
        """Privileged oracle helper returning exact workpiece ground truth."""
        entry_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "workshop_target_hole_entry"
        )
        seated_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "workshop_target_hole_seated_tip"
        )
        if entry_site < 0 or seated_site < 0:
            raise RuntimeError("Workshop target-hole landmarks are missing")
        return {
            "workpiece_id": "workshop_frame_joint",
            "fixture_center_world_m": list(WORKSHOP_FRAME_FIXTURE_CENTER),
            "hole_entry_center_world_m": self.data.site_xpos[entry_site].tolist(),
            "seated_fastener_tip_world_m": self.data.site_xpos[seated_site].tolist(),
            "hole_axis_world": [0.0, 0.0, 1.0],
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
                cab_reg["target_world_m"] = [-0.75, 0.56, 0.83]
                cab_reg["rig_position_world_m"] = [-0.75, -0.70, 1.22]
                cab_reg["inspection_volume"]["minimum_world_m"] = [-0.75, 0.35, 0.65]
                cab_reg["inspection_volume"]["maximum_world_m"] = [-0.25, 0.75, 1.12]
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
        mechanism = WORKSHOP_CONTAINER_JOINTS.get(region_id)
        if mechanism is None:
            raise ValueError(f"Unknown workshop region: {region_id}")
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mechanism["actuator"]
        )
        if actuator_id < 0:
            raise RuntimeError(f"Missing actuator: {mechanism['actuator']}")
        return actuator_id

    def open_container(self, region_id: str, steps: int = 1000) -> dict[str, Any]:
        """Open one storage region with the KitchenScene articulation contract.

        The public action is simply ``OPEN(region_id)``.  The mechanism is
        chosen from the argument: both drawers use the duplicated kitchen
        pull-out, while the cabinet uses its own hinged door.
        """
        if region_id not in WORKSHOP_REGIONS:
            raise ValueError(f"Unknown storage container: {region_id}")
        if self.state.container_open_state[region_id]:
            return {"region_id": region_id, "opened": True, "newly_opened": False}

        mechanism = WORKSHOP_CONTAINER_JOINTS[region_id]
        actuator_id = self._container_actuator_id(region_id)
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, mechanism["joint"]
        )
        if joint_id < 0:
            raise RuntimeError(f"Missing storage joint: {mechanism['joint']}")
        target = float(mechanism["open_val"])
        self.data.ctrl[actuator_id] = target
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        joint_position = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        if joint_position < 0.80 * target:
            raise RuntimeError(
                f"{region_id} failed to reach its physical open target: "
                f"joint={joint_position:.4f}, target={target:.4f}"
            )
        was_new = region_id not in self.state.opened_containers
        self.state.container_open_state[region_id] = True
        self.state.opened_containers.add(region_id)
        return {
            "region_id": region_id,
            "opened": True,
            "newly_opened": was_new,
        }

    def close_container(self, region_id: str, steps: int = 600) -> None:
        """Maintenance-only reset helper; it is intentionally not a Workshop action."""
        if region_id not in WORKSHOP_REGIONS:
            raise ValueError(f"Unknown workshop region: {region_id}")
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

    def launch_viewer(
        self,
        camera: str = FREE_CAMERA,
        *,
        actions_panel: bool = True,
    ) -> None:
        if camera != FREE_CAMERA and camera not in WORKSHOP_CAMERAS:
            raise ValueError(f"Unknown workshop camera: {camera}")
        if self.robot_name == ROBOT_GOOGLE and actions_panel:
            from .workshop_actions import launch_workshop_action_viewer

            launch_workshop_action_viewer(self, camera)
            return
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
    parser.add_argument("--variant", default="F0_MANUAL_FIRST_ONE_REGION", help="Workshop variant to initialize.")
    parser.add_argument("--list-variants", action="store_true", help="List all available variants.")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--no-actions-panel",
        action="store_true",
        help="Open only the passive MuJoCo viewer without Workshop controls.",
    )
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
        scene.launch_viewer(
            arguments.camera,
            actions_panel=not arguments.no_actions_panel,
        )


if __name__ == "__main__":
    main()
