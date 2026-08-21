"""
Kitchen Scene Loader for 'Reasoning Before Planning' MuJoCo Experiments.

Dynamically loads the kitchen_base.xml, injects objects into containers
and onto the countertop based on a YAML scene config, and provides
an API for opening/closing containers and querying visibility.
"""

import yaml
import copy
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter
from importlib.util import find_spec
from pathlib import Path
from dataclasses import dataclass, field
import random

import mujoco
import mujoco.viewer
import numpy as np


# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
CONFIGS_DIR = ROOT / "configs"
KITCHEN_BASE = ASSETS_DIR / "kitchen_base.xml"
OBJECT_LIB = ASSETS_DIR / "objects" / "object_library.xml"
OBJECT_MESHES_DIR = ASSETS_DIR / "objects" / "meshes"
SCENE_CONFIGS = CONFIGS_DIR / "scene_configs.yaml"
KITCHEN_FEASIBILITY_VARIANTS = (
    CONFIGS_DIR / "kitchen_feasibility_variants.yaml"
)

ROBOT_FETCH = "fetch"
ROBOT_GOOGLE = "google"
ROBOT_NONE = "none"
# Google Robot is the sole interactive robot backend on the integration
# branch.  Fetch composition remains internal temporarily so historical
# motion tests and old saved experiments can still be audited, but it is not
# exposed by the production CLI or selected by default.
ROBOT_CHOICES = (ROBOT_GOOGLE, ROBOT_NONE)


# ── Fetch mobile manipulator ─────────────────────────────────────────────
# The visual/collision meshes and kinematic tree come from the maintained
# Gymnasium-Robotics Fetch assets. We adapt its benchmark-fixed base to three
# controllable planar joints and use position actuators for the arm/gripper.
FETCH_PACKAGE = "gymnasium_robotics"
FETCH_ASSET_SUBDIR = Path("envs") / "assets" / "fetch"
FETCH_BASE_POSE = {
    # Start behind the centered serving table with symmetric left/right routes
    # to the workstation manipulation poses.
    "pos": "0 -1.10 0",
    # Fetch's local +X (arm-forward direction) faces world +Y toward the table.
    "quat": "0.7071068 0 0 0.7071068",
}

FETCH_HOME_QPOS = {
    "robot0:base_forward_joint": 0.0,
    "robot0:base_lateral_joint": 0.0,
    "robot0:base_yaw_joint": 0.0,
    "robot0:torso_lift_joint": 0.20,
    "robot0:head_pan_joint": 0.0,
    "robot0:head_tilt_joint": 0.35,
    # Relaxed navigation pose: the arm is compactly tucked against the torso,
    # leaving the head camera and both side-navigation corridors unobstructed.
    "robot0:shoulder_pan_joint": 1.32,
    "robot0:shoulder_lift_joint": 1.40,
    "robot0:upperarm_roll_joint": -0.20,
    "robot0:elbow_flex_joint": 1.72,
    "robot0:forearm_roll_joint": 0.0,
    "robot0:wrist_flex_joint": 1.66,
    "robot0:wrist_roll_joint": 0.0,
    "robot0:r_gripper_finger_joint": 0.035,
    "robot0:l_gripper_finger_joint": 0.035,
}

FETCH_ACTUATORS = (
    # name, joint, kp, ctrl_min, ctrl_max
    ("robot0:base_forward_actuator", "robot0:base_forward_joint", 6000, -1.0, 1.0),
    ("robot0:base_lateral_actuator", "robot0:base_lateral_joint", 6000, -1.5, 1.5),
    ("robot0:base_yaw_actuator", "robot0:base_yaw_joint", 3500, -3.14, 3.14),
    ("robot0:torso_lift_actuator", "robot0:torso_lift_joint", 3000, 0.0386, 0.3861),
    ("robot0:head_pan_actuator", "robot0:head_pan_joint", 100, -1.57, 1.57),
    ("robot0:head_tilt_actuator", "robot0:head_tilt_joint", 100, -0.76, 1.45),
    ("robot0:shoulder_pan_actuator", "robot0:shoulder_pan_joint", 650, -1.6056, 1.6056),
    ("robot0:shoulder_lift_actuator", "robot0:shoulder_lift_joint", 650, -1.221, 1.518),
    ("robot0:upperarm_roll_actuator", "robot0:upperarm_roll_joint", 350, -3.14, 3.14),
    ("robot0:elbow_flex_actuator", "robot0:elbow_flex_joint", 650, -2.251, 2.251),
    ("robot0:forearm_roll_actuator", "robot0:forearm_roll_joint", 350, -3.14, 3.14),
    ("robot0:wrist_flex_actuator", "robot0:wrist_flex_joint", 350, -2.16, 2.16),
    ("robot0:wrist_roll_actuator", "robot0:wrist_roll_joint", 250, -3.14, 3.14),
    ("robot0:r_gripper_finger_actuator", "robot0:r_gripper_finger_joint", 1000, 0.0, 0.05),
    ("robot0:l_gripper_finger_actuator", "robot0:l_gripper_finger_joint", 1000, 0.0, 0.05),
)


# ── Google Robot mobile manipulator ─────────────────────────────────────
# The kinematic tree and meshes come from MuJoCo Menagerie. Its published
# model fixes the base, so the kitchen adapter adds the same ideal planar
# planning joints used by Fetch. Menagerie itself remains outside this Git
# repository and is located through MUJOCO_MENAGERIE_PATH or the workspace's
# default third_party checkout.
GOOGLE_BASE_POSE = {
    "pos": "0 -1.25 0.06205",
    # Google Robot's local +X faces world +Y toward the workstation.
    "quat": "0.7071068 0 0 0.7071068",
}

GOOGLE_HOME_QPOS = {
    "google:base_forward_joint": 0.0,
    "google:base_lateral_joint": 0.0,
    "google:base_yaw_joint": 0.0,
    # The zero arm configuration is a compact, upright navigation pose.
    "google:joint_torso": 0.0,
    "google:joint_shoulder": 0.0,
    "google:joint_bicep": 0.0,
    "google:joint_elbow": 0.0,
    "google:joint_forearm": 0.0,
    "google:joint_wrist": 0.0,
    "google:joint_gripper": 0.0,
    "google:joint_finger_right": 0.35,
    "google:joint_finger_left": 0.35,
}

GOOGLE_ACTUATORS = (
    # name, joint, kp, ctrl_min, ctrl_max
    ("google:base_forward_actuator", "google:base_forward_joint", 6000, -1.0, 1.25),
    ("google:base_lateral_actuator", "google:base_lateral_joint", 6000, -1.5, 1.5),
    ("google:base_yaw_actuator", "google:base_yaw_joint", 3500, -3.14, 3.14),
    # The upstream gains are intended for free-space model demonstration.
    # Kitchen contact tasks need tighter tracking so a commanded bilateral
    # grasp does not settle centimetres off-centre under arm and finger load.
    ("google:joint_torso_actuator", "google:joint_torso", 120, -4.49, 1.35),
    ("google:joint_shoulder_actuator", "google:joint_shoulder", 120, -2.66, 3.18),
    ("google:joint_bicep_actuator", "google:joint_bicep", 100, -2.13, 3.71),
    ("google:joint_elbow_actuator", "google:joint_elbow", 100, -2.05, 3.79),
    ("google:joint_forearm_actuator", "google:joint_forearm", 80, -2.92, 2.92),
    ("google:joint_wrist_actuator", "google:joint_wrist", 80, -1.79, 1.79),
    ("google:joint_gripper_actuator", "google:joint_gripper", 60, -4.49, 1.35),
    ("google:joint_finger_right_actuator", "google:joint_finger_right", 60, 0.01, 1.3),
    ("google:joint_finger_left_actuator", "google:joint_finger_left", 60, 0.01, 1.3),
)

GOOGLE_FORCE_RANGES = {
    "google:joint_torso": "-150 150",
    "google:joint_shoulder": "-150 150",
    "google:joint_bicep": "-30 30",
    "google:joint_elbow": "-30 30",
    "google:joint_forearm": "-30 30",
    "google:joint_wrist": "-30 30",
    "google:joint_gripper": "-30 30",
    "google:joint_finger_right": "-30 30",
    "google:joint_finger_left": "-30 30",
}

ROBOT_HOME_QPOS = {
    ROBOT_FETCH: FETCH_HOME_QPOS,
    ROBOT_GOOGLE: GOOGLE_HOME_QPOS,
}
ROBOT_ACTUATORS = {
    ROBOT_FETCH: FETCH_ACTUATORS,
    ROBOT_GOOGLE: GOOGLE_ACTUATORS,
}
ROBOT_BASE_JOINTS = {
    ROBOT_FETCH: (
        "robot0:base_forward_joint",
        "robot0:base_lateral_joint",
        "robot0:base_yaw_joint",
    ),
    ROBOT_GOOGLE: (
        "google:base_forward_joint",
        "google:base_lateral_joint",
        "google:base_yaw_joint",
    ),
}


# ── physically supported object placement ────────────────────────────────
# Slot Z values identify the top of a support surface, not the object centre.
# The loader adds the object's lowest-point offset so objects start just above
# shelves/trays instead of intersecting them.
OBJECT_SUPPORT_HEIGHT = {
    "mug": 0.04065, "cup": 0.03076, "glass": 0.055,
    "plate": 0.01336, "small_plate": 0.00735,
    "bowl": 0.02750, "mixing_bowl": 0.02750,
    "spoon": 0.01045, "oversized_spoon": 0.01463,
    "fork": 0.00773, "knife": 0.00762, "marker": 0.00945,
    "stirrer": 0.003, "spatula": 0.006, "tongs": 0.005,
    "kettle": 0.06817, "coffee_jar": 0.09287, "sugar_jar": 0.06724,
    "coffee_can": 0.07009, "sugar_box": 0.08802,
    "milk_carton": 0.060, "tea_box": 0.01944, "bread": 0.025,
    "butter": 0.015, "jam_jar": 0.040, "napkin": 0.003,
    "biscuits": 0.020, "pot_with_soup": 0.03303,
    "gso_canister_distractor": 0.07187,
    "gso_spatula_distractor": 0.01234,
    "ab3_narrow_deep_cup": 0.04768,
    "ab3_medium_deep_mug": 0.05488,
    "ab3_shallow_bowl": 0.02613,
    "ab3_deep_bowl": 0.04263,
    "ab3_short_narrow_spoon": 0.01045,
    "ab3_medium_spoon": 0.01045,
    "ab3_long_wide_spoon": 0.01150,
    "ab3_long_narrow_spoon": 0.01045,
    "ab3_partial_spoon": 0.01045,
    "ab3_long_narrow_fork": 0.00773,
    "s1i_wide_shallow_cup": 0.03537,
    "s1i_narrow_deep_bowl": 0.04263,
    "s1i_soup_spoon": 0.01045,
    "s1i_coffee_near_miss_spoon": 0.01097,
    "s1i_final_long_narrow_spoon": 0.01045,
    "s1i_oversized_spoon": 0.01150,
    "s1i_c2_soup_spoon": 0.01045,
    "s1i_compact_kettle": 0.05317,
    "s1i_compact_coffee_jar": 0.07244,
    "feas_coffee_small_shallow_cup": 0.04306,
    "feas_coffee_medium_deep_mug": 0.05660,
    "feas_coffee_extra_deep_mug": 0.09000,
    "feas_coffee_wide_very_deep_cup": 0.07227,
    "feas_c2_medium_spoon": 0.01045,
    "feas_soup_wide_shallow_bowl": 0.02475,
    "feas_soup_narrow_shallow_bowl": 0.02475,
    "feas_soup_wide_deep_bowl": 0.06325,
    "feas_narrow_short_spoon": 0.01045,
    "feas_narrow_medium_spoon": 0.01045,
    "feas_medium_spoon": 0.01045,
    "feas_wide_long_spoon": 0.01045,
    "feas_soup_wide_short_spoon": 0.01045,
    "feas_soup_wide_medium_spoon": 0.01045,
    "feas_semantic_decoy_spoon": 0.01045,
}

# Controlled visible-geometry variants for Ablation 3.  These declarations
# are consumed only while constructing the rendered scene; runtime semantic
# and geometric inference receives RGB-D, masks, and calibration—not this
# mapping, its scale factors, or the source object names.
SCENE_OBJECT_VARIANTS = {
    "ab3_narrow_deep_cup": {
        "base": "cup", "scale": (1.15, 1.15, 1.55),
        "mesh": "mesh_ab3_narrow_deep_cup",
        "material": "mat_s1i_cup_cream",
    },
    "ab3_medium_deep_mug": {
        "base": "mug", "scale": (0.92, 0.92, 1.35),
        "mesh": "mesh_ab3_medium_deep_mug",
        "material": "mat_s1i_mug_blue",
    },
    "ab3_shallow_bowl": {
        "base": "bowl", "scale": (0.90, 0.90, 0.95),
        "mesh": "mesh_ab3_shallow_bowl",
        "material": "mat_s1i_bowl_ivory",
    },
    "ab3_deep_bowl": {
        "base": "bowl", "scale": (0.92, 0.92, 1.55),
        "mesh": "mesh_ab3_deep_bowl",
        "material": "mat_s1i_bowl_blue",
    },
    "ab3_short_narrow_spoon": {
        "base": "spoon", "scale": (0.70, 0.75, 1.0),
        "mesh": "mesh_ab3_short_narrow_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "ab3_medium_spoon": {
        "base": "spoon", "scale": (0.95, 1.45, 1.0),
        "mesh": "mesh_ab3_medium_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "ab3_long_wide_spoon": {
        "base": "spoon", "scale": (1.22, 1.55, 1.10),
        "mesh": "mesh_ab3_long_wide_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "ab3_long_narrow_spoon": {
        "base": "spoon", "scale": (1.25, 0.95, 1.0),
        "mesh": "mesh_ab3_long_narrow_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "ab3_partial_spoon": {
        "base": "spoon", "scale": (0.72, 0.85, 1.0),
        "mesh": "mesh_ab3_partial_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "ab3_long_narrow_fork": {
        # A 24 cm serving fork remains physically plausible while preserving
        # enough tine pixels for multi-view RGB recognition at the calibrated
        # inspection distance.
        "base": "fork", "scale": (1.35, 1.0, 1.0),
        "mesh": "mesh_ab3_long_narrow_fork",
    },
    # Integrated Scene 1 variants reuse the same source meshes while keeping
    # the scene-design namespace independent from Ablation 3. These scales
    # affect rendered evidence only and are never visible to inference.
    "s1i_wide_shallow_cup": {
        "base": "cup", "scale": (1.30, 1.30, 1.15),
        "mesh": "mesh_s1i_wide_shallow_cup",
        "material": "mat_s1i_cup_sage",
    },
    "s1i_narrow_deep_bowl": {
        "base": "bowl", "scale": (0.95, 0.95, 1.55),
        "mesh": "mesh_s1i_narrow_deep_bowl",
        "material": "mat_s1i_bowl_sage",
    },
    "s1i_soup_spoon": {
        "base": "spoon", "scale": (0.90, 0.90, 1.0),
        "mesh": "mesh_s1i_soup_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "s1i_coffee_near_miss_spoon": {
        "base": "spoon", "scale": (1.20, 1.40, 1.05),
        "mesh": "mesh_s1i_coffee_near_miss_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "s1i_final_long_narrow_spoon": {
        "base": "spoon", "scale": (1.32, 0.95, 1.0),
        "mesh": "mesh_s1i_final_long_narrow_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "s1i_oversized_spoon": {
        "base": "spoon", "scale": (1.35, 1.70, 1.10),
        "mesh": "mesh_s1i_oversized_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "s1i_c2_soup_spoon": {
        "base": "spoon", "scale": (1.15, 1.15, 1.0),
        "mesh": "mesh_s1i_c2_soup_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "s1i_compact_kettle": {
        "base": "kettle", "scale": (0.78, 0.78, 0.78),
        "mesh": "mesh_s1i_compact_kettle",
    },
    "s1i_compact_coffee_jar": {
        "base": "coffee_jar", "scale": (0.78, 0.78, 0.78),
        "mesh": "mesh_s1i_compact_coffee_jar",
    },
    "feas_coffee_small_shallow_cup": {
        "base": "cup", "scale": (0.82, 0.82, 1.40),
        "mesh": "mesh_feas_coffee_small_shallow_cup",
        "material": "mat_s1i_cup_cream",
    },
    "feas_coffee_medium_deep_mug": {
        "base": "cup", "scale": (1.05, 1.05, 1.84),
        "mesh": "mesh_feas_coffee_medium_deep_mug",
        "material": "mat_s1i_mug_blue",
    },
    "feas_coffee_extra_deep_mug": {
        "base": "cup", "scale": (1.60, 1.60, 3.0),
        "mesh": "mesh_feas_coffee_extra_deep_mug",
        "material": "mat_s1i_mug_blue",
    },
    "feas_coffee_wide_very_deep_cup": {
        "base": "cup", "scale": (1.25, 1.25, 2.10),
        "mesh": "mesh_feas_coffee_wide_very_deep_cup",
        "material": "mat_s1i_cup_sage",
    },
    "feas_c2_medium_spoon": {
        "base": "spoon", "scale": (0.90, 1.15, 1.0),
        "mesh": "mesh_feas_c2_medium_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "feas_soup_wide_shallow_bowl": {
        "base": "bowl", "scale": (1.05, 1.05, 0.90),
        "mesh": "mesh_feas_soup_wide_shallow_bowl",
        "material": "mat_s1i_bowl_ivory",
    },
    "feas_soup_narrow_shallow_bowl": {
        "base": "bowl", "scale": (0.45, 0.45, 0.90),
        "mesh": "mesh_feas_soup_narrow_shallow_bowl",
        "material": "mat_s1i_bowl_blue",
    },
    "feas_soup_wide_deep_bowl": {
        "base": "bowl", "scale": (1.05, 1.05, 3.20),
        "mesh": "mesh_feas_soup_wide_deep_bowl",
        "material": "mat_s1i_bowl_sage",
    },
    "feas_narrow_short_spoon": {
        "base": "spoon", "scale": (0.55, 0.70, 1.0),
        "mesh": "mesh_feas_narrow_short_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "feas_narrow_medium_spoon": {
        "base": "spoon", "scale": (0.85, 0.70, 1.0),
        "mesh": "mesh_feas_narrow_medium_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "feas_medium_spoon": {
        "base": "spoon", "scale": (0.95, 1.65, 1.0),
        "mesh": "mesh_feas_medium_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "feas_wide_long_spoon": {
        "base": "spoon", "scale": (1.35, 2.20, 1.0),
        "mesh": "mesh_feas_wide_long_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "feas_soup_wide_short_spoon": {
        "base": "spoon", "scale": (0.90, 1.70, 1.0),
        "mesh": "mesh_feas_soup_wide_short_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "feas_soup_wide_medium_spoon": {
        "base": "spoon", "scale": (1.15, 2.20, 1.0),
        "mesh": "mesh_feas_soup_wide_medium_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
    "feas_semantic_decoy_spoon": {
        "base": "spoon", "scale": (0.55, 4.20, 1.0),
        "mesh": "mesh_feas_semantic_decoy_spoon",
        "material": "mat_s1i_spoon_inspection",
    },
}

UTENSIL_OBJECTS = {
    "spoon", "oversized_spoon", "fork", "knife", "stirrer", "spatula", "tongs",
    "gso_spatula_distractor",
    "ab3_short_narrow_spoon", "ab3_medium_spoon",
    "ab3_long_wide_spoon", "ab3_long_narrow_spoon",
    "ab3_partial_spoon",
    "ab3_long_narrow_fork",
    "s1i_soup_spoon", "s1i_coffee_near_miss_spoon",
    "s1i_final_long_narrow_spoon",
    "s1i_oversized_spoon",
    "s1i_c2_soup_spoon",
    "feas_c2_medium_spoon",
    "feas_narrow_short_spoon", "feas_narrow_medium_spoon",
    "feas_medium_spoon", "feas_wide_long_spoon",
    "feas_soup_wide_short_spoon", "feas_soup_wide_medium_spoon",
    "feas_semantic_decoy_spoon",
}
CENTRED_DRAWER_OBJECTS = {
    "spoon", "oversized_spoon", "fork", "knife", "stirrer", "tongs",
    "gso_spatula_distractor",
    "ab3_short_narrow_spoon", "ab3_medium_spoon",
    "ab3_long_wide_spoon", "ab3_long_narrow_spoon",
    "ab3_partial_spoon",
    "ab3_long_narrow_fork",
    "s1i_soup_spoon", "s1i_coffee_near_miss_spoon",
    "s1i_final_long_narrow_spoon",
    "s1i_oversized_spoon",
    "s1i_c2_soup_spoon",
    "feas_c2_medium_spoon",
    "feas_narrow_short_spoon", "feas_narrow_medium_spoon",
    "feas_medium_spoon", "feas_wide_long_spoon",
    "feas_soup_wide_short_spoon", "feas_soup_wide_medium_spoon",
    "feas_semantic_decoy_spoon",
}

ACTION_PICK_OBJECTS = {
    "kettle", "coffee_jar", "sugar_jar", "spoon",
    "fork", "knife", "stirrer", "spatula", "tongs", "napkin",
    "gso_spatula_distractor",
}
PASSIVE_HANDLE_OBJECTS = {
    "spoon", "fork", "knife", "stirrer", "spatula", "tongs",
    "gso_spatula_distractor",
}

# Scene-construction fixtures transport free-jointed drawer objects with their
# moving trays. They are released immediately after direct opening and before
# the inspection rig captures evidence.
STORAGE_FIXTURE_EQUALITIES = {
    "D1": "storage_fixture_D1_oversized_spoon",
    "D2": "storage_fixture_D2_ablation3_utensil",
    "C2": "storage_fixture_C2_upright_spoon",
}

# Positions are RELATIVE to the container body origin: (x, y, support_z).
# C1 alternates between its lower floor and shelf. C2 reserves the entire
# lower level for a large plate and puts smaller objects on the shelf.
CONTAINER_SLOTS = {
    "C1": {
        "parent_body": "cabinet_C1",
        "slots": [
            (-0.10, 0.0, -0.205),
            (-0.10, 0.0, -0.032),
            (0.10, 0.0, -0.032),
            (0.10, 0.0, -0.205),
        ],
    },
    "C2": {
        "parent_body": "cabinet_C2",
        "slots": [
            (0.0, 0.0, -0.205),
            (-0.10, 0.0, -0.032),
            (0.10, 0.0, -0.032),
            (0.0, -0.06, -0.032),
        ],
    },
    "D1": {
        "parent_body": "drawer_D1_tray",
        "slots": [
            (-0.09, 0.030, -0.052),
            (-0.09, -0.030, -0.052),
            (-0.09, -0.090, -0.052),
            (-0.09, 0.075, -0.052),
        ],
    },
    "D2": {
        "parent_body": "drawer_D2_tray",
        "slots": [
            (-0.09, 0.030, -0.052),
            (-0.09, -0.030, -0.052),
            (-0.09, -0.090, -0.052),
            (-0.09, 0.075, -0.052),
        ],
    },
    "B1": {
        "parent_body": "box_B1",
        "slots": [
            (-0.070, 0.0, 0.003),
            (0.070, 0.0, 0.003),
            (0.0, 0.040, 0.003),
        ],
    },
}

# Countertop spots contain (x, y, support_z) world coordinates.
COUNTER_SPOTS = {
    # Negative Y is the Fetch side of the worktop. Keep the primary
    # ingredients away from the upper cabinets and within an easy frontal
    # gripper approach corridor.
    "counter_spot_1": (-0.35, -0.32, 0.580),
    "counter_spot_2": (-0.15, -0.30, 0.580),
    "counter_spot_3": (0.05, -0.30, 0.580),
    "counter_spot_4": (0.20, -0.30, 0.580),
    "counter_spot_5": (0.25, -0.34, 0.580),
    "counter_spot_6": (0.0, -0.22, 0.580),
    # Ablation 2 uses a wider two-row layout so repeated cup, bowl, and
    # utensil instances remain visually separable in all five RGB views.
    "counter_spot_7": (-0.50, -0.34, 0.580),
    "counter_spot_8": (-0.34, -0.34, 0.580),
    "counter_spot_9": (-0.13, -0.28, 0.580),
    "counter_spot_10": (0.08, -0.28, 0.580),
    "counter_spot_11": (0.26, -0.11, 0.580),
    "counter_spot_12": (0.31, -0.34, 0.580),
    # Ablation 3 separates four containers (back row) from four candidate
    # utensils (front row) while keeping every instance inside INITIAL's
    # calibrated inspection volume.
    "counter_spot_13": (-0.55, -0.34, 0.580),
    "counter_spot_14": (-0.35, -0.34, 0.580),
    "counter_spot_15": (-0.10, -0.32, 0.580),
    "counter_spot_16": (0.16, -0.32, 0.580),
    "counter_spot_17": (-0.52, -0.10, 0.580),
    "counter_spot_18": (-0.25, -0.10, 0.580),
    "counter_spot_19": (0.02, -0.10, 0.580),
    "counter_spot_20": (0.29, -0.10, 0.580),
    # Integrated stress-test layout. Six target vessels occupy the rear two
    # rows; flat candidates and compact clutter use the front row. All points
    # remain inside INITIAL's calibrated region gate.
    "counter_spot_21": (-0.52, -0.39, 0.580),
    "counter_spot_22": (-0.34, -0.39, 0.580),
    "counter_spot_23": (-0.08, -0.39, 0.580),
    "counter_spot_24": (-0.48, -0.23, 0.580),
    "counter_spot_25": (-0.20, -0.23, 0.580),
    "counter_spot_26": (0.08, -0.23, 0.580),
    "counter_spot_27": (-0.32, -0.22, 0.580),
    "counter_spot_28": (-0.07, -0.21, 0.580),
    "counter_spot_29": (0.18, -0.21, 0.580),
    "counter_spot_30": (0.36, -0.21, 0.580),
    "counter_spot_31": (-0.57, -0.075, 0.580),
    "counter_spot_32": (-0.35, -0.075, 0.580),
    "counter_spot_33": (-0.13, -0.075, 0.580),
    "counter_spot_34": (0.09, -0.075, 0.580),
    "counter_spot_35": (0.31, -0.075, 0.580),
    # Preparation sources for the final integrated task. They occupy a
    # separate right-hand column inside the calibrated INITIAL gate so they
    # are observed without altering the closed-storage utensil benchmark.
    "counter_spot_37": (0.30, -0.36, 0.580),
    "counter_spot_38": (0.50, -0.26, 0.580),
    "counter_spot_39": (0.46, -0.10, 0.580),
    "counter_spot_40": (0.12, -0.075, 0.580),
    # Authoritative integrated-scene layout revision. The vessel and utensil
    # rows are separated by 22 cm. Long utensils are spaced by their planar
    # footprint rather than by an arbitrary centre-to-centre interval. This
    # remains collision-free for every randomized assignment of the six task
    # vessels to the three visible slots.
    "counter_spot_41": (-0.60, -0.08, 0.580),
    "counter_spot_42": (-0.37, -0.08, 0.580),
    "counter_spot_43": (-0.14, -0.08, 0.580),
    "counter_spot_44": (-0.37, -0.08, 0.580),
    "counter_spot_45": (-0.14, -0.08, 0.580),
    "counter_spot_46": (0.09, -0.08, 0.580),
    "counter_spot_47": (-0.62, -0.30, 0.580),
    "counter_spot_48": (-0.40, -0.30, 0.580),
    "counter_spot_49": (-0.10, -0.30, 0.580),
    "counter_spot_50": (0.15, -0.29, 0.580),
    "counter_spot_51": (0.27, -0.29, 0.580),
    "counter_spot_52": (0.48, -0.31, 0.580),
    "counter_spot_53": (0.49, -0.08, 0.580),
    "counter_spot_54": (0.42, -0.055, 0.580),
    # Phase-B F1 physical-validity correction.  The former spot 16 placed the
    # shallow and narrow bowls with overlapping collision shells.  This nearby
    # point preserves the same support, visibility, and workspace while adding
    # only the clearance required for independent physical extraction.
    "counter_spot_55": (0.00, -0.43, 0.580),
    # Phase-B F1 source-access correction. The kettle stays on the same visible
    # countertop but leaves a collision-free side-grasp corridor to the coffee
    # source. Fresh Phase-1 evidence is required after this correction.
    # Keeps the kettle clear of both the bowl extraction corridor and the
    # coffee-jar side grasp while remaining inside the original observation
    # association gate used by the frozen F1 evidence.
    "counter_spot_57": (0.30, -0.31, 0.580),
    # Isolated semantic-counterexample location at the far left of INITIAL's
    # calibrated volume. It keeps a thin marker clear of vessel silhouettes.
    "counter_spot_36": (-0.67, -0.10, 0.580),
}

# The kettle handle is deliberately grasped rather than approximating a pick
# through the pot centre. Bring that handle into the arm's exact vertical-IK
# workspace while preserving a clear gap to the coffee jar beside it.
COUNTERTOP_OBJECT_OFFSETS = {
    "kettle": (0.06, 0.0, 0.0),
}

CAMERAS = (
    "left_shoulder_camera",
    "right_shoulder_camera",
    "overhead_camera",
    "side_camera",
    "wrist_camera",
    "front_camera",
)
ROBOT_CAMERAS = ("head_camera_rgb",)
CAMERA_CHOICES = CAMERAS + ROBOT_CAMERAS
FREE_CAMERA = "free"
VIEW_CAMERA_CHOICES = (FREE_CAMERA,) + CAMERA_CHOICES

# Joint/actuator names for each container
CONTAINER_JOINTS = {
    "C1": {"joint": "C1_door_joint", "actuator": "C1_door_actuator", "open_val": 1.4},
    "C2": {"joint": "C2_door_joint", "actuator": "C2_door_actuator", "open_val": 1.4},
    "D1": {"joint": "D1_slide_joint", "actuator": "D1_slide_actuator", "open_val": 0.25},
    "D2": {"joint": "D2_slide_joint", "actuator": "D2_slide_actuator", "open_val": 0.25},
    "B1": {"joint": "B1_lid_joint", "actuator": "B1_lid_actuator", "open_val": 1.8},
}


@dataclass
class SceneConfig:
    """Parsed scene configuration."""
    name: str
    goal: str
    countertop_objects: dict          # {spot_name: object_name}
    container_contents: dict          # {container_id: [object_names]}
    required_objects: list
    substitution_map: dict
    optimal_search_order: list
    optimal_inspections: int
    notes: str = ""


INTEGRATED_PRIMARY_SCENE = "S1_integrated_kitchen_object_function_primary"
INTEGRATED_SCENE_PREFIX = "S1_integrated_kitchen_object_function"


def is_integrated_kitchen_scene(scene_name: str) -> bool:
    return scene_name.startswith(INTEGRATED_SCENE_PREFIX)
INTEGRATED_TARGET_VESSELS = (
    "ab3_narrow_deep_cup",
    "ab3_medium_deep_mug",
    "s1i_wide_shallow_cup",
    "ab3_shallow_bowl",
    "ab3_deep_bowl",
    "s1i_narrow_deep_bowl",
)
INTEGRATED_VISIBLE_TARGET_SPOTS = (
    "counter_spot_41", "counter_spot_44", "counter_spot_45"
)
INTEGRATED_HIDDEN_TARGET_REGIONS = ("C2", "B1", "C1")

# Conservative top-view half extents (x, y), in metres, after the integrated
# scene's visual scaling and countertop orientation. These values are used
# only to reject bad scene layouts; perception and geometry inference never
# receive them. The fork is crosswise in this scene, hence its long y extent.
INTEGRATED_COUNTERTOP_HALF_EXTENTS = {
    "ab3_narrow_deep_cup": (0.060, 0.060),
    "ab3_medium_deep_mug": (0.070, 0.060),
    "s1i_wide_shallow_cup": (0.070, 0.070),
    "ab3_shallow_bowl": (0.080, 0.080),
    "ab3_deep_bowl": (0.080, 0.080),
    "s1i_narrow_deep_bowl": (0.075, 0.075),
    "ab3_short_narrow_spoon": (0.085, 0.025),
    "ab3_medium_spoon": (0.110, 0.030),
    "ab3_long_wide_spoon": (0.165, 0.045),
    "ab3_long_narrow_fork": (0.025, 0.140),
    "marker": (0.018, 0.060),
    "s1i_compact_kettle": (0.080, 0.080),
    "s1i_compact_coffee_jar": (0.045, 0.045),
}
INTEGRATED_COUNTERTOP_BUFFER_M = 0.015


def validate_integrated_countertop_clearance(config: SceneConfig) -> None:
    """Reject overlapping primary-scene countertop placements.

    The check is deterministic and conservative. It runs after seeded target
    randomization, so every accepted seed has at least 15 mm of axis-aligned
    separation between the configured object footprints.
    """
    if config.name != INTEGRATED_PRIMARY_SCENE:
        return
    placed = []
    for spot, object_kind in config.countertop_objects.items():
        half_extent = INTEGRATED_COUNTERTOP_HALF_EXTENTS.get(object_kind)
        if half_extent is None:
            raise ValueError(
                f"Missing integrated countertop footprint for {object_kind}"
            )
        x, y, _z = COUNTER_SPOTS[spot]
        placed.append((spot, object_kind, x, y, *half_extent))
    for index, first in enumerate(placed):
        for second in placed[index + 1:]:
            x_overlap = (
                abs(first[2] - second[2])
                < first[4] + second[4] + INTEGRATED_COUNTERTOP_BUFFER_M
            )
            y_overlap = (
                abs(first[3] - second[3])
                < first[5] + second[5] + INTEGRATED_COUNTERTOP_BUFFER_M
            )
            if x_overlap and y_overlap:
                raise ValueError(
                    "Integrated countertop objects violate the 15 mm buffer: "
                    f"{first[0]}={first[1]} and {second[0]}={second[1]}"
                )


def configure_integrated_target_layout(
    config: SceneConfig,
    layout_seed: int | None,
) -> dict:
    """Apply a capacity-safe target layout and return its public manifest.

    The YAML layout is the deterministic default. A seed shuffles only the
    six task vessels: three remain on the counter and one is placed in each
    cupboard/box region. Tool candidates and the fixed inspection order never
    move, so layout variation does not silently change the benchmark policy.
    """
    if config.name != INTEGRATED_PRIMARY_SCENE:
        if layout_seed is not None:
            raise ValueError(
                "--layout-seed is supported only for "
                f"{INTEGRATED_PRIMARY_SCENE}"
            )
        return {"mode": "configured", "seed": None}

    if layout_seed is not None:
        rng = random.Random(layout_seed)
        coffee_vessels = list(INTEGRATED_TARGET_VESSELS[:3])
        soup_vessels = list(INTEGRATED_TARGET_VESSELS[3:])
        rng.shuffle(coffee_vessels)
        rng.shuffle(soup_vessels)
        # Closed-region placements are restricted to scan/rig combinations
        # that provide reliable RGB-D evidence: C2 gets a profile-rich coffee
        # vessel, while B1 and C1 each get a bowl. Exact identities still
        # shuffle with the seed. The narrow handle-free cup remains visible.
        c2_candidates = [
            vessel for vessel in coffee_vessels
            if vessel != "ab3_narrow_deep_cup"
        ]
        rng.shuffle(c2_candidates)
        c2_target = c2_candidates[0]
        coffee_vessels.remove(c2_target)
        b1_target = soup_vessels.pop(0)
        c1_target = soup_vessels.pop(0)
        remaining = [*coffee_vessels, *soup_vessels]
        rng.shuffle(remaining)
        hidden_by_region = {
            "C2": c2_target, "B1": b1_target, "C1": c1_target,
        }
        visible = remaining
        for spot, vessel in zip(INTEGRATED_VISIBLE_TARGET_SPOTS, visible):
            config.countertop_objects[spot] = vessel
        for region, vessel in hidden_by_region.items():
            non_targets = [
                item for item in config.container_contents[region]
                if item not in INTEGRATED_TARGET_VESSELS
            ]
            config.container_contents[region] = [*non_targets, vessel]

    target_locations = {}
    for spot, item in config.countertop_objects.items():
        if item in INTEGRATED_TARGET_VESSELS:
            target_locations[item] = "INITIAL"
    for region, items in config.container_contents.items():
        for item in items:
            if item in INTEGRATED_TARGET_VESSELS:
                target_locations[item] = region

    if set(target_locations) != set(INTEGRATED_TARGET_VESSELS):
        raise ValueError("Integrated target layout must place every vessel once")
    if any(len(items) > 2 for items in config.container_contents.values()):
        raise ValueError("Integrated closed regions may contain at most two objects")
    for region in INTEGRATED_HIDDEN_TARGET_REGIONS:
        count = sum(
            item in INTEGRATED_TARGET_VESSELS
            for item in config.container_contents[region]
        )
        if count != 1:
            raise ValueError(f"{region} must contain exactly one target vessel")
    validate_integrated_countertop_clearance(config)
    return {
        "mode": "seeded_random" if layout_seed is not None else "configured",
        "seed": layout_seed,
        "target_locations": target_locations,
        "closed_region_contents": {
            region: list(items)
            for region, items in config.container_contents.items()
        },
        "capacity_policy": {
            "maximum_objects_per_closed_region": 2,
            "maximum_target_vessels_per_cupboard_or_box": 1,
        },
    }


@dataclass
class SceneState:
    """Runtime state tracking visibility and search progress."""
    opened_containers: set = field(default_factory=set)
    visible_objects: set = field(default_factory=set)       # Objects currently visible
    visible_object_counts: Counter = field(default_factory=Counter)
    hidden_objects: dict = field(default_factory=dict)      # {container: [objects]} - ground truth
    found_in: dict = field(default_factory=dict)            # {object: container_found_in}
    object_positions: dict = field(default_factory=dict)    # {object: (x, y, z)}
    container_open_state: dict = field(default_factory=dict)


def load_all_configs() -> dict[str, SceneConfig]:
    """Load all scene configurations from YAML."""
    with open(SCENE_CONFIGS) as f:
        raw = yaml.safe_load(f)

    configs = {}
    for name, cfg in raw.get("scenes", {}).items():
        sc = SceneConfig(
            name=name,
            goal=cfg["goal"],
            countertop_objects=cfg.get("countertop_objects", {}),
            container_contents=cfg.get("container_contents", {}),
            required_objects=cfg.get("required_objects", []),
            substitution_map=cfg.get("substitution_map", {}),
            optimal_search_order=cfg.get("optimal_search_order", []),
            optimal_inspections=cfg.get("optimal_inspections", 0),
            notes=cfg.get("notes", ""),
        )
        configs[name] = sc
    if KITCHEN_FEASIBILITY_VARIANTS.exists():
        benchmark = yaml.safe_load(
            KITCHEN_FEASIBILITY_VARIANTS.read_text(encoding="utf-8")
        ) or {}
        expected_goal = benchmark.get("goal_instruction")
        for variant_id, variant in benchmark.get("variants", {}).items():
            scene_name = variant["scene_name"]
            base_name = variant["base_scene"]
            if base_name not in configs:
                raise ValueError(
                    f"Feasibility variant {variant_id} has unknown base "
                    f"scene {base_name}"
                )
            derived = copy.deepcopy(configs[base_name])
            derived.name = scene_name
            if expected_goal is not None and derived.goal != expected_goal:
                raise ValueError(
                    f"Feasibility variant {variant_id} changed the goal"
                )
            if "countertop_objects" in variant:
                derived.countertop_objects = dict(
                    variant["countertop_objects"]
                )
            else:
                for spot in variant.get("countertop_remove", []):
                    derived.countertop_objects.pop(spot, None)
                derived.countertop_objects.update(
                    variant.get("countertop_set", {})
                )
            if "container_contents" in variant:
                derived.container_contents = {
                    region: list(items)
                    for region, items in variant[
                        "container_contents"
                    ].items()
                }
            derived.optimal_search_order = list(
                benchmark.get(
                    "inspection_order", derived.optimal_search_order
                )
            )
            derived.optimal_inspections = int(
                variant.get(
                    "expected_inspections",
                    len(derived.optimal_search_order),
                )
            )
            derived.notes = (
                f"Controlled task-feasibility benchmark variant {variant_id}. "
                + str(variant.get("description", ""))
            )
            configs[scene_name] = derived
    return configs


def _parse_object_library() -> tuple[dict[str, ET.Element], list[ET.Element]]:
    """Parse reusable object bodies and their visual asset declarations."""
    tree = ET.parse(OBJECT_LIB)
    root = tree.getroot()
    objects = {}
    for body in root.findall("body"):
        objects[body.get("name")] = body
    asset_root = root.find("asset")
    assets = list(asset_root) if asset_root is not None else []
    return objects, assets


def _fetch_asset_dir() -> Path:
    """Return the Fetch asset directory installed by Gymnasium-Robotics."""
    package_spec = find_spec(FETCH_PACKAGE)
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError(
            "Fetch robot assets are unavailable. Install dependencies with "
            "`pip install -r mujoco_scenes/requirements.txt` or use the Docker image."
        )

    package_dir = Path(next(iter(package_spec.submodule_search_locations))).resolve()
    asset_dir = package_dir / FETCH_ASSET_SUBDIR
    required = (asset_dir / "shared.xml", asset_dir / "robot.xml")
    if not all(path.exists() for path in required):
        raise RuntimeError(f"Incomplete Fetch model installation at: {asset_dir}")
    return asset_dir


def _selected_robot(include_robot: bool, robot: str | None) -> str:
    """Resolve the Google/no-robot selector while preserving legacy loading."""
    selected = robot if robot is not None else (
        ROBOT_GOOGLE if include_robot else ROBOT_NONE
    )
    if selected not in ROBOT_CHOICES:
        choices = ", ".join(ROBOT_CHOICES)
        raise ValueError(f"Unknown robot '{selected}'. Choose from: {choices}")
    return selected


def _google_robot_dir() -> Path:
    """Return the external MuJoCo Menagerie Google Robot directory."""
    configured = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.name != "google_robot":
            candidate = candidate / "google_robot"
    else:
        candidate = (
            ROOT.parent.parent
            / "third_party"
            / "mujoco_menagerie"
            / "google_robot"
        ).resolve()

    required = (candidate / "robot.xml", candidate / "assets")
    if not all(path.exists() for path in required):
        raise RuntimeError(
            "Google Robot assets are unavailable. Sparse-clone MuJoCo "
            "Menagerie into `../third_party/mujoco_menagerie` or set "
            "MUJOCO_MENAGERIE_PATH to the Menagerie checkout."
        )
    return candidate


def _remove_named_body(parent: ET.Element, body_name: str) -> None:
    """Remove a direct child body if present."""
    for body in list(parent.findall("body")):
        if body.get("name") == body_name:
            parent.remove(body)


def _inject_fetch_robot(root: ET.Element, fetch_dir: Path) -> None:
    """Merge and adapt Farama's Fetch MJCF into the kitchen model."""
    shared_root = ET.parse(fetch_dir / "shared.xml").getroot()
    robot_root = ET.parse(fetch_dir / "robot.xml").getroot()

    asset = root.find("asset")
    default = root.find("default")
    worldbody = root.find("worldbody")
    contact = root.find("contact")
    actuator = root.find("actuator")

    # Only Fetch-specific materials and meshes are required. The benchmark's
    # skybox, floor, table and block materials belong to its original task.
    shared_asset = shared_root.find("asset")
    for element in shared_asset:
        name = element.get("name", "")
        if element.tag == "mesh" or name.startswith("robot0:"):
            asset.append(copy.deepcopy(element))

    fetch_defaults = shared_root.find("default/default")
    default.append(copy.deepcopy(fetch_defaults))

    for exclusion in shared_root.findall("contact/exclude"):
        contact.append(copy.deepcopy(exclusion))

    robot_body = None
    for body in robot_root.findall("body"):
        if body.get("name") == "robot0:base_link":
            robot_body = copy.deepcopy(body)
            break
    if robot_body is None:
        raise RuntimeError("Fetch robot.xml does not contain robot0:base_link")

    robot_body.set("pos", FETCH_BASE_POSE["pos"])
    robot_body.set("quat", FETCH_BASE_POSE["quat"])

    # Keep the high-detail base mesh visual-only and use a smooth collision
    # proxy. condim=1 removes artificial floor drag from the holonomic planar
    # joints while retaining normal collision with furniture and obstacles.
    base_visual = robot_body.find("geom[@name='robot0:base_link']")
    if base_visual is not None:
        base_visual.set("contype", "0")
        base_visual.set("conaffinity", "0")
        base_visual.set("group", "1")
    ET.SubElement(
        robot_body,
        "geom",
        {
            "name": "robot0:base_collision_proxy",
            "type": "cylinder",
            "size": "0.27 0.18",
            "pos": "0 0 0.18",
            "condim": "1",
            "priority": "2",
            "friction": "0 0 0",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "0 0 0 0",
            "group": "3",
        },
    )

    # Convert the benchmark's locked XYZ base into a planar mobile base.
    base_joints = robot_body.findall("joint")[:3]
    if len(base_joints) != 3:
        raise RuntimeError("Unexpected Fetch base joint layout")
    joint_specs = (
        ("robot0:base_forward_joint", "slide", "1 0 0", "-1 1", "500"),
        ("robot0:base_lateral_joint", "slide", "0 1 0", "-1.5 1.5", "500"),
        ("robot0:base_yaw_joint", "hinge", "0 0 1", "-3.14 3.14", "100"),
    )
    for joint, (name, joint_type, axis, joint_range, damping) in zip(base_joints, joint_specs):
        joint.attrib.clear()
        joint.set("name", name)
        joint.set("type", joint_type)
        joint.set("axis", axis)
        joint.set("range", joint_range)
        joint.set("limited", "true")
        joint.set("damping", damping)
        joint.set("armature", "0.1")

    # Remove task-only external camera. The head camera stays, and the original
    # gripper RGB camera becomes the requested wrist camera.
    _remove_named_body(robot_body, "robot0:external_camera_body_0")
    for camera in robot_body.iter("camera"):
        if camera.get("name") == "gripper_camera_rgb":
            camera.set("name", "robot0:gripper_camera_rgb_legacy")

    gripper_body = None
    for body in robot_body.iter("body"):
        if body.get("name") == "robot0:gripper_link":
            gripper_body = body
            break
    if gripper_body is None:
        raise RuntimeError("Fetch gripper body missing from robot.xml")

    # Keep the stock finger boxes as visuals, but use collision fingertips
    # that end 10 mm beyond robot0:grip instead of 18.5 mm. The original long
    # collision boxes make a vertical pinch of a flat utensil mathematically
    # require penetrating the tabletop.
    for side, lateral in (("r", -0.008), ("l", 0.008)):
        finger_name = f"robot0:{side}_gripper_finger_link"
        finger_body = next(
            body for body in gripper_body.iter("body")
            if body.get("name") == finger_name
        )
        visual = finger_body.find(f"geom[@name='{finger_name}']")
        if visual is None:
            raise RuntimeError(f"Fetch finger geometry missing: {finger_name}")
        visual.set("name", f"{finger_name}_visual")
        visual.set("contype", "0")
        visual.set("conaffinity", "0")
        ET.SubElement(
            finger_body,
            "geom",
            {
                "name": finger_name,
                "type": "box",
                "pos": f"-0.00425 {lateral} 0",
                "size": "0.03425 0.007 0.0135",
                "condim": "4",
                "friction": "1 0.05 0.01",
                "rgba": "0 0 0 0",
                "group": "3",
            },
        )
    ET.SubElement(
        gripper_body,
        "camera",
        {
            "name": "wrist_camera",
            "pos": "0.03 0 0.04",
            # Look forward and slightly down from the home gripper pose.
            "xyaxes": "0 -1 0 -0.7936 0 0.6085",
            "fovy": "65",
        },
    )

    # Gravity compensation keeps the initial arm pose stable until a planner
    # starts commanding the joint-space position actuators.
    for body in robot_body.iter("body"):
        if body.find("inertial") is not None:
            body.set("gravcomp", "1")
    for joint in robot_body.iter("joint"):
        name = joint.get("name", "")
        if name.startswith("robot0:") and "base_" not in name:
            if "torso_lift" in name:
                damping, armature = "50", "1"
            elif "gripper_finger" in name:
                damping, armature = "20", "0.2"
            elif "head_" in name:
                damping, armature = "5", "0.1"
            else:
                damping, armature = "10", "0.5"
            joint.set("damping", damping)
            joint.set("armature", armature)
            joint.attrib.pop("stiffness", None)

    # The kitchen's temporary wrist camera is replaced by the real Fetch wrist
    # camera at the gripper link.
    _remove_named_body(worldbody, "wrist_camera_mount")
    worldbody.append(robot_body)

    for name, joint, kp, ctrl_min, ctrl_max in FETCH_ACTUATORS:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": joint,
                "kp": str(kp),
                "ctrllimited": "true",
                "ctrlrange": f"{ctrl_min} {ctrl_max}",
            },
        )


def _prefix_google_robot(robot_root: ET.Element) -> None:
    """Namespace Menagerie identifiers and default classes in-place."""
    class_names = {
        "robot": "google_robot",
        "visual": "google_visual",
        "collision": "google_collision",
        "finger_base": "google_finger_base",
        "finger_tip": "google_finger_tip",
    }
    for element in robot_root.iter():
        for attribute in ("class", "childclass"):
            value = element.get(attribute)
            if value in class_names:
                element.set(attribute, class_names[value])

    asset = robot_root.find("asset")
    mesh_names = {}
    texture_names = {}
    material_names = {}
    if asset is not None:
        for element in asset:
            if element.tag == "mesh":
                old_name = element.get("name") or Path(element.get("file")).stem
                new_name = f"google:{old_name}"
                mesh_names[old_name] = new_name
                element.set("name", new_name)
            elif element.tag == "texture":
                old_name = element.get("name")
                new_name = f"google:{old_name}"
                texture_names[old_name] = new_name
                element.set("name", new_name)
            elif element.tag == "material":
                old_name = element.get("name")
                new_name = f"google:{old_name}"
                material_names[old_name] = new_name
                element.set("name", new_name)

        for material in asset.findall("material"):
            texture = material.get("texture")
            if texture in texture_names:
                material.set("texture", texture_names[texture])

    for element in robot_root.iter():
        if element.tag in {"body", "joint", "site"}:
            name = element.get("name")
            if name:
                element.set("name", f"google:{name}")
        elif element.tag == "geom":
            name = element.get("name")
            if name:
                element.set("name", f"google:{name}")

        mesh = element.get("mesh")
        if mesh in mesh_names:
            element.set("mesh", mesh_names[mesh])
        material = element.get("material")
        if material in material_names:
            element.set("material", material_names[material])


def _inject_google_robot(
    root: ET.Element,
    google_dir: Path,
    base_pose: dict[str, str] | None = None,
) -> None:
    """Merge and adapt Menagerie's Google Robot into the kitchen model."""
    robot_root = ET.parse(google_dir / "robot.xml").getroot()
    _prefix_google_robot(robot_root)

    asset = root.find("asset")
    default = root.find("default")
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")

    for element in robot_root.findall("asset/*"):
        asset.append(copy.deepcopy(element))
    for element in robot_root.findall("default/default"):
        default.append(copy.deepcopy(element))

    robot_body = robot_root.find("worldbody/body[@name='google:base_link']")
    if robot_body is None:
        raise RuntimeError("Menagerie Google Robot base_link body is missing")
    robot_body = copy.deepcopy(robot_body)
    resolved_base_pose = GOOGLE_BASE_POSE if base_pose is None else base_pose
    robot_body.set("pos", resolved_base_pose["pos"])
    robot_body.set("quat", resolved_base_pose["quat"])

    joint_specs = (
        ("google:base_forward_joint", "slide", "1 0 0", "-1 1.25", "750"),
        ("google:base_lateral_joint", "slide", "0 1 0", "-1.5 1.5", "750"),
        ("google:base_yaw_joint", "hinge", "0 0 1", "-3.14 3.14", "100"),
    )
    for index, (name, joint_type, axis, joint_range, damping) in enumerate(
        joint_specs
    ):
        robot_body.insert(
            index,
            ET.Element(
                "joint",
                {
                    "name": name,
                    "type": joint_type,
                    "axis": axis,
                    "range": joint_range,
                    "limited": "true",
                    "damping": damping,
                    "armature": "0.1",
                },
            ),
        )

    # Detailed base and wheel meshes are visual-only in the ideal holonomic
    # abstraction. A smooth proxy prevents artificial wheel/floor drag while
    # retaining collision checks against the workstation and serving table.
    for geom in robot_body.findall("geom"):
        if geom.get("class") == "google_collision":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("rgba", "0 0 0 0")
    ET.SubElement(
        robot_body,
        "geom",
        {
            "name": "google:base_collision_proxy",
            "type": "cylinder",
            "size": "0.27 0.15",
            "pos": "0 0 0.15",
            "condim": "1",
            "priority": "2",
            "friction": "0 0 0",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "0 0 0 0",
            "group": "3",
        },
    )

    ET.SubElement(
        robot_body,
        "camera",
        {
            "name": "head_camera_rgb",
            # Sit ahead of the fixed head shell rather than inside its mesh.
            "pos": "0.22 0 1.30",
            # Look along local +X and 15 degrees down toward the worktop.
            "xyaxes": "0 -1 0 0.258819 0 0.965926",
            "fovy": "65",
        },
    )
    gripper_body = next(
        (
            body
            for body in robot_body.iter("body")
            if body.get("name") == "google:link_gripper"
        ),
        None,
    )
    if gripper_body is None:
        raise RuntimeError("Menagerie Google Robot gripper body is missing")

    # Menagerie's fingertip collision geoms are intentionally anonymous.  The
    # kitchen picker needs stable left/right names so it can require bilateral
    # object contact instead of treating palm or table contact as a grasp.
    for side in ("right", "left"):
        tip_body = next(
            (
                body
                for body in gripper_body.iter("body")
                if body.get("name") == f"google:link_finger_tip_{side}"
            ),
            None,
        )
        if tip_body is None:
            raise RuntimeError(f"Menagerie Google Robot {side} fingertip is missing")
        collision_geoms = [
            geom
            for geom in tip_body.findall("geom")
            if geom.get("class") != "google_visual"
        ]
        if len(collision_geoms) != 6:
            raise RuntimeError(
                f"Expected 6 Google {side} fingertip collision geoms, "
                f"found {len(collision_geoms)}"
            )
        for index, geom in enumerate(collision_geoms):
            geom.set("name", f"google:{side}_finger_pad_{index}")

    ET.SubElement(
        gripper_body,
        "camera",
        {
            "name": "wrist_camera",
            "pos": "0 0 0.16",
            # The gripper and its public site extend along local +Z.
            "xyaxes": "1 0 0 0 -1 0",
            "fovy": "65",
        },
    )

    _remove_named_body(worldbody, "wrist_camera_mount")
    worldbody.append(robot_body)

    for name, joint, kp, ctrl_min, ctrl_max in GOOGLE_ACTUATORS:
        attributes = {
            "name": name,
            "joint": joint,
            "kp": str(kp),
            "ctrllimited": "true",
            "ctrlrange": f"{ctrl_min} {ctrl_max}",
        }
        force_range = GOOGLE_FORCE_RANGES.get(joint)
        if force_range:
            attributes["forcelimited"] = "true"
            attributes["forcerange"] = force_range
        ET.SubElement(actuator, "position", attributes)


def _load_fetch_binary_assets(fetch_dir: Path) -> dict[str, bytes]:
    """Load mesh/texture bytes for MjModel.from_xml_string()."""
    supported = {".stl", ".obj", ".png", ".jpg", ".jpeg"}
    source_dirs = (fetch_dir, fetch_dir.parent / "stls" / "fetch")
    assets = {}
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in source_dir.iterdir():
            if path.is_file() and path.suffix.lower() in supported:
                assets[path.name] = path.read_bytes()
    return assets


def _load_google_binary_assets(google_dir: Path) -> dict[str, bytes]:
    """Load Menagerie Google Robot meshes and textures for MJCF compile."""
    supported = {".stl", ".obj", ".png", ".jpg", ".jpeg"}
    assets = {}
    for path in (google_dir / "assets").iterdir():
        if path.is_file() and path.suffix.lower() in supported:
            assets[path.name] = path.read_bytes()
    return assets


def _load_object_binary_assets() -> dict[str, bytes]:
    """Load prepared kitchen OBJ and texture files for in-memory MJCF compile."""
    supported = {".obj", ".png", ".jpg", ".jpeg"}
    assets = {}
    for path in OBJECT_MESHES_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            key = path.relative_to(OBJECT_MESHES_DIR).as_posix()
            assets[key] = path.read_bytes()
    return assets


def build_scene_xml(
    config: SceneConfig,
    include_robot: bool = True,
    robot: str | None = None,
) -> str:
    """
    Take kitchen_base.xml, inject objects from the scene config, and return
    the complete XML string ready for mujoco.MjModel.from_xml_string().
    """
    robot_name = _selected_robot(include_robot, robot)
    tree = ET.parse(KITCHEN_BASE)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    obj_lib, object_assets = _parse_object_library()
    asset_root = root.find("asset")
    for element in object_assets:
        asset_root.append(copy.deepcopy(element))

    if robot_name == ROBOT_FETCH:
        _inject_fetch_robot(root, _fetch_asset_dir())
    elif robot_name == ROBOT_GOOGLE:
        _inject_google_robot(root, _google_robot_dir())

    # Track unique instance counters for duplicates
    instance_count = {}
    object_instances: list[tuple[str, str]] = []

    def _scale_variant_body(
        body: ET.Element,
        scale: tuple[float, float, float],
        mesh_name: str,
        material_name: str | None = None,
    ) -> None:
        """Scale rendered/proxy geometry for scene construction only."""

        scale_vector = np.asarray(scale, dtype=float)

        def scaled_values(text: str, factors: np.ndarray) -> str:
            values = np.fromstring(text, sep=" ")
            if len(values) != len(factors):
                return text
            return " ".join(
                f"{value:.8g}" for value in values * factors
            )

        for element in body.iter():
            if element.get("pos"):
                element.set(
                    "pos",
                    scaled_values(element.get("pos"), scale_vector),
                )
            if element.tag != "geom":
                continue
            geom_type = element.get("type", "sphere")
            if (
                material_name is not None
                and element.get("class") == "visual"
                and element.get("material") not in {
                    "mat_tomato_soup", "mat_hot_water", "mat_steam",
                    "mat_coffee_powder", "mat_open_jar_rim",
                }
            ):
                # Keep analytic interior surfaces visually consistent with
                # the variant mesh. Previously the shell received the clean
                # material while the cup/bowl floor retained a stretched YCB
                # texture, producing the conspicuous mismatched interiors.
                element.set("material", material_name)
            if geom_type == "mesh":
                element.set("mesh", mesh_name)
                continue
            if element.get("fromto"):
                element.set(
                    "fromto",
                    scaled_values(
                        element.get("fromto"),
                        np.concatenate((scale_vector, scale_vector)),
                    ),
                )
            if not element.get("size"):
                continue
            values = np.fromstring(element.get("size"), sep=" ")
            if geom_type in {"box", "ellipsoid"} and len(values) == 3:
                factors = scale_vector
            elif geom_type in {"cylinder", "capsule"} and len(values) == 2:
                factors = np.asarray(
                    [max(scale_vector[:2]), scale_vector[2]]
                )
            elif geom_type in {"sphere", "capsule"} and len(values) == 1:
                factors = np.asarray([max(scale_vector[1:])])
            else:
                continue
            element.set(
                "size",
                " ".join(
                    f"{value:.8g}" for value in values * factors
                ),
            )

    def _get_instance_name(obj_name: str) -> str:
        """Generate unique name for duplicate objects (e.g., spoon → spoon_2)."""
        if obj_name not in instance_count:
            instance_count[obj_name] = 0
        instance_count[obj_name] += 1
        count = instance_count[obj_name]
        if count == 1:
            return obj_name
        return f"{obj_name}_{count}"

    def _inject_object(
        obj_name: str,
        support_pos: tuple,
        quat: str | None = None,
        support_height_override: float | None = None,
    ):
        """Add an object body directly to worldbody at the given position."""
        variant = SCENE_OBJECT_VARIANTS.get(obj_name)
        library_name = variant["base"] if variant else obj_name
        if library_name not in obj_lib:
            print(f"  [WARNING] Object '{obj_name}' not found in object library, skipping.")
            return

        instance_name = _get_instance_name(obj_name)
        obj_elem = copy.deepcopy(obj_lib[library_name])
        if variant:
            _scale_variant_body(
                obj_elem,
                variant["scale"],
                variant["mesh"],
                variant.get("material"),
            )

        # Convert a support-surface location into a non-penetrating body centre.
        support_height = (
            support_height_override
            if support_height_override is not None
            else OBJECT_SUPPORT_HEIGHT.get(obj_name, 0.03)
        )
        pos = (support_pos[0], support_pos[1],
               support_pos[2] + support_height + 0.002)
        obj_elem.set("pos", f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
        if quat is not None:
            obj_elem.set("quat", quat)

        # Rename body and all children to instance name if different
        if instance_name != library_name:
            old_name = library_name
            new_name = instance_name
            for child in obj_elem.iter():
                if child is obj_elem:
                    continue
                name = child.get("name")
                if name:
                    if old_name in name:
                        child.set("name", name.replace(old_name, new_name))
                    else:
                        # Decorative children such as liquid/powder surfaces
                        # do not necessarily contain the library body name.
                        # They still occupy MuJoCo's global geom/site namespace
                        # and therefore need an instance-qualified name.
                        child.set("name", f"{new_name}__{name}")
                joint = child.get("joint")
                if joint and old_name in joint:
                    child.set("joint", joint.replace(old_name, new_name))
            # Rename the root exactly once. Iterating over the already-renamed
            # root used to turn the second instance into ``cup_2_2`` while
            # persistent discovery correctly looked for ``cup_2``.
            obj_elem.set("name", new_name)

        worldbody.append(obj_elem)
        object_instances.append((instance_name, obj_name))
        return instance_name

    def _get_body_world_pos(body_name: str) -> np.ndarray:
        """Trace path from target body up to worldbody to get absolute world position."""
        parent_map = {}
        for parent in root.iter():
            for child in parent:
                if child.tag == "body":
                    parent_map[child] = parent

        # Find target body element
        target_elem = None
        for body in root.iter("body"):
            if body.get("name") == body_name:
                target_elem = body
                break

        if target_elem is None:
            return np.zeros(3)

        pos_sum = np.zeros(3)
        curr = target_elem
        while curr is not None and curr.tag == "body":
            pos_str = curr.get("pos", "0 0 0")
            pos_val = np.fromstring(pos_str, sep=" ")
            if len(pos_val) == 3:
                pos_sum += pos_val
            curr = parent_map.get(curr)
        return pos_sum

    # ── Place countertop objects ──────────────────────────────────────────
    for spot, obj_name in config.countertop_objects.items():
        if spot in COUNTER_SPOTS:
            pos = np.asarray(COUNTER_SPOTS[spot]) + np.asarray(
                COUNTERTOP_OBJECT_OFFSETS.get(obj_name, (0.0, 0.0, 0.0))
            )
            countertop_quat = None
            if (
                config.name
                == "S1_integrated_kitchen_object_function_feasibility_F1"
                and spot == "counter_spot_18"
                and obj_name == "s1i_final_long_narrow_spoon"
            ):
                # Keep the selected duplicate's far-handle grasp corridor
                # clear of the adjacent deep bowl. The object centre, support,
                # identity, dimensions, and functional assignment are
                # unchanged; only its in-plane presentation is reversed.
                countertop_quat = "0 0 0 1"
            elif (
                config.name == "S1_joint_stir_initial_preference"
                and obj_name in {"fork", "ab3_long_narrow_fork"}
            ):
                # Present the normal fork orthogonally to the neighbouring
                # spoon so its tines remain separable in the initial RGB
                # views. This is scene construction, never runtime inference.
                countertop_quat = "0.7071068 0 0 0.7071068"
            elif (
                (
                    config.name.startswith("S1_ablation3_multi_target")
                    or config.name.startswith(
                        "S1_integrated_kitchen_object_function"
                    )
                )
                and obj_name in {"fork", "ab3_long_narrow_fork"}
            ):
                # Expose the fork tines crosswise so the open-vocabulary RGB
                # detector can distinguish them from a spoon silhouette.
                countertop_quat = "0.7071068 0 0 0.7071068"
            _inject_object(obj_name, pos, quat=countertop_quat)
        else:
            print(f"  [WARNING] Unknown counter spot '{spot}', skipping {obj_name}.")

    # ── Place container objects ───────────────────────────────────────────
    storage_fixture_instances: dict[str, list[str]] = {}
    for container_id, objects in config.container_contents.items():
        if container_id not in CONTAINER_SLOTS:
            print(f"  [WARNING] Unknown container '{container_id}', skipping.")
            continue

        cinfo = CONTAINER_SLOTS[container_id]
        parent_body_pos = _get_body_world_pos(cinfo["parent_body"])
        slots = cinfo["slots"]

        # Real-scale drinkware needs size-aware allocation. C1 has roughly
        # 16 cm below its enlarged shelf, so taller objects go above. C2 stacks
        # plate-like objects on its lower level and places other items above.
        if container_id == "C1" and is_integrated_kitchen_scene(config.name):
            # Keep the searched utensil and randomized vessel on separate
            # shelves. This respects per-shelf capacity and prevents the
            # vessel silhouette from suppressing thin-tool RGB detections.
            allocated_slots = [slots[0], slots[2]]
            if config.name == "S1_integrated_kitchen_object_function_primary":
                # Keep the production-selected bowl in C1's already validated
                # lower-left extraction corridor.  The non-selected spoon is
                # placed on the separate upper-right shelf, preserving both
                # identities, region membership, visibility, and capacity.
                allocated_slots = [slots[2], slots[0]]
        elif container_id == "C1":
            lower_slots = [slots[0], slots[3]]
            shelf_slots = [slots[1], slots[2]]
            allocated_slots = []
            for obj_name in objects:
                is_tall = 2 * OBJECT_SUPPORT_HEIGHT.get(obj_name, 0.03) > 0.15
                preferred = shelf_slots if is_tall else lower_slots
                fallback = lower_slots if is_tall else shelf_slots
                allocated_slots.append((preferred or fallback).pop(0))
        elif container_id == "C2" and is_integrated_kitchen_scene(config.name):
            # Keep the second integrated object away from the cabinet side
            # wall. The centred rear slot gives the calibrated C2 rig a
            # complete rim/interior view for tall drinkware.
            allocated_slots = [slots[0], slots[3]]
        elif container_id == "C2":
            plate_objects = {"plate", "small_plate"}
            nonplate_count = sum(obj not in plate_objects for obj in objects)
            shelf_x = {
                0: [],
                1: [0.0],
                2: [-0.090, 0.110],
                3: [-0.120, 0.0, 0.120],
            }[nonplate_count]
            plate_stack_support = -0.205
            shelf_index = 0
            allocated_slots = []
            for obj_name in objects:
                if obj_name in plate_objects:
                    allocated_slots.append((0.0, 0.0, plate_stack_support))
                    plate_stack_support += (
                        2 * OBJECT_SUPPORT_HEIGHT[obj_name] + 0.002
                    )
                else:
                    allocated_slots.append((shelf_x[shelf_index], 0.0, -0.032))
                    shelf_index += 1
        else:
            allocated_slots = slots

        for i, obj_name in enumerate(objects):
            if i >= len(allocated_slots):
                print(f"  [WARNING] Container {container_id} full, cannot place {obj_name}.")
                continue
            slot_rel_pos = np.array(allocated_slots[i], dtype=float)
            if (
                is_integrated_kitchen_scene(config.name)
                and container_id == "C2"
                and obj_name != "s1i_c2_soup_spoon"
            ):
                # Reserve the left half of C2's shelf for the vessel so the
                # upright utensil has a separate, unobstructed central-right
                # grasp corridor.
                slot_rel_pos[0] = -0.10
            # Centred scanned utensils use the drawer middle lane; legacy +X
            # primitives retain the left lane. Compact objects use the right
            # lane so realistic-width napkins/boxes cannot overlap utensils.
            if container_id in {"D1", "D2"} and obj_name not in UTENSIL_OBJECTS:
                slot_rel_pos[0] = 0.08
            elif container_id == "D1" and obj_name == "oversized_spoon":
                # Centre the deliberately long counterexample across D1's
                # usable width. This is physical scene placement only; no
                # runtime inference consumes the configured offset.
                slot_rel_pos[0] = -0.06
            elif container_id in {"D1", "D2"} and obj_name in CENTRED_DRAWER_OBJECTS:
                # The low vertical IK corridor is centred around X=±0.39.
                # Bias each mirrored utensil lane three centimetres outward;
                # after D1's 180-degree yaw this puts both logical handle ends
                # on those symmetric reachable lines while preserving wall
                # clearance for the longest scanned tool.
                slot_rel_pos[0] = -0.03 if container_id == "D1" else 0.03
                if container_id == "D2" and obj_name == "tongs":
                    # Keep the two-pronged tongs clear of a co-located
                    # fixture-held scanned utensil.  Their previous adjacent
                    # lane could begin in contact after D2 opened, eventually
                    # driving the free joint unstable and invalidating the
                    # following C2 RGB-D capture.
                    slot_rel_pos[1] = -0.09
            world_pos = parent_body_pos + np.array(slot_rel_pos)
            if (
                is_integrated_kitchen_scene(config.name)
                and container_id == "C2"
                and obj_name == "s1i_c2_soup_spoon"
            ):
                # Store the spoon fully inside C2, standing on its bowl and
                # leaning gently toward the right wall.  Keep the lean small
                # enough that the handle tip and the horizontal gripper
                # corridor remain inside the cabinet side panels.  A
                # temporary weld stabilizes this presentation until the
                # contact-confirmed Phase B pick releases it.  The broad
                # side-wall buffer centres both finger links in the opening,
                # not merely the spoon collision proxy.
                # Pivot the presentation about its shelf contact and shift it
                # only slightly right.  The bowl touches C2's right panel,
                # while the middle handle stays clear for the horizontal
                # gripper approach.
                world_pos[0] = parent_body_pos[0] + 0.1138
                # Present the handle near the open front while retaining a
                # positive buffer behind the cabinet face.  This keeps the
                # straight +Y approach short and away from both side panels.
                world_pos[1] = parent_body_pos[1] - 0.11
                # Raise the complete collision/visual envelope above the
                # shelf.  The spoon mesh origin is near its handle rather
                # than its lowest point, so the generic support-height rule
                # otherwise lets its bowl pass through the shelf.
                # Calibrated from the transformed collision envelope: this
                # puts the handle tip on C2's shelf top (rather than leaving
                # the visually obvious 2.7 mm air gap).
                # Object positions serialize at 0.1 mm precision, so use the
                # next representable height above exact contact.  This leaves
                # a sub-millimetre numerical clearance with no visible float.
                world_pos[2] = parent_body_pos[2] - 0.0095
                injected_instance = _inject_object(
                    obj_name,
                    world_pos,
                    quat="0.3007058 0 0.9537169 0",
                    support_height_override=0.050,
                )
            elif container_id == "B1" and obj_name == "coffee_jar":
                # The coffee jar remains slightly taller than B1's closed
                # interior. Store it centred on its side along the longer X
                # dimension. The shorter sugar jar now fits upright in the
                # enlarged box and is deliberately left upright so it cannot
                # roll indefinitely on its cylindrical collision proxy.
                world_pos[0] = parent_body_pos[0]
                horizontal_radius = 0.03962
                injected_instance = _inject_object(
                    obj_name,
                    world_pos,
                    quat="0.7071068 0 0.7071068 0",
                    support_height_override=horizontal_radius,
                )
            elif (
                is_integrated_kitchen_scene(config.name)
                and container_id in {"C1", "C2", "B1"}
                and obj_name in UTENSIL_OBJECTS
            ):
                # A slight diagonal exposes both the spoon bowl and handle to
                # more than one region-facing RGB camera without changing any
                # inference threshold or object dimensions.
                injected_instance = _inject_object(
                    obj_name, world_pos, quat="0.9914449 0 0 0.1305262"
                )
            elif container_id == "D1" and obj_name in UTENSIL_OBJECTS:
                # Point the logical handle end toward the robot centreline.
                # D2's default -X handle direction already points inward; D1
                # is its mirror and therefore needs a 180-degree yaw. This
                # keeps the requested end grasp inside the accurate central
                # IK workspace instead of reaching past the drawer's far wall.
                injected_instance = _inject_object(
                    obj_name, world_pos, quat="0 0 0 1"
                )
            else:
                injected_instance = _inject_object(obj_name, world_pos)
            if injected_instance is not None and (
                container_id in {"D1", "D2"}
                or (
                    container_id == "C2"
                    and obj_name == "s1i_c2_soup_spoon"
                )
            ):
                # Free-jointed objects live in world coordinates, not beneath
                # the translating tray body. Weld every drawer item during
                # deterministic direct opening; otherwise compact objects can
                # remain under the counter or destabilize against the moving
                # tray after only the large counterexample is transported.
                storage_fixture_instances.setdefault(container_id, []).append(
                    injected_instance
                )

    # Contact-confirmed pick actions can enable one of these initially
    # inactive welds after both gripper fingers touch a supported object.
    # The relative pose is filled from the live state when the grasp occurs,
    # so enabling a weld never snaps an object to a pre-recorded pose.
    if storage_fixture_instances:
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        for region_id, instance_names in sorted(
            storage_fixture_instances.items()
        ):
            for fixture_index, instance_name in enumerate(instance_names):
                fixture_name = (
                    STORAGE_FIXTURE_EQUALITIES[region_id]
                    if fixture_index == 0
                    else f"{STORAGE_FIXTURE_EQUALITIES[region_id]}_{fixture_index}"
                )
                fixture_attributes = {
                    "name": fixture_name,
                    "body1": CONTAINER_SLOTS[region_id]["parent_body"],
                    "body2": instance_name,
                    "active": "true",
                    "solref": "0.002 1",
                }
                ET.SubElement(equality, "weld", fixture_attributes)

    if robot_name in {ROBOT_FETCH, ROBOT_GOOGLE}:
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        equality_prefix = "google" if robot_name == ROBOT_GOOGLE else "robot0"
        gripper_body_name = (
            "google:link_gripper"
            if robot_name == ROBOT_GOOGLE
            else "robot0:gripper_link"
        )
        supported_objects = {
            "kettle", "coffee_jar", "sugar_jar", "spoon", "fork",
            "mug", "cup", "bowl",
        }
        for instance_name, object_kind in object_instances:
            execution_base_kind = SCENE_OBJECT_VARIANTS.get(
                object_kind, {}
            ).get("base", object_kind)
            if execution_base_kind not in supported_objects:
                continue
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"{equality_prefix}:pick_weld_{instance_name}",
                    "body1": gripper_body_name,
                    "body2": instance_name,
                    "active": "false",
                    "solref": "0.01 1",
                },
            )
            if execution_base_kind in PASSIVE_HANDLE_OBJECTS:
                # Activated after the initial vertical lift. Unlike the
                # transport weld, a connect equality fixes only the handle
                # pinch point and leaves all three rotational DOFs free, so
                # gravity can swing the working end naturally below the hand.
                ET.SubElement(
                    equality,
                    "connect",
                    {
                        "name": f"{equality_prefix}:pick_pivot_{instance_name}",
                        "body1": gripper_body_name,
                        "body2": instance_name,
                        "anchor": "0 0 0",
                        "active": "false",
                        "solref": "0.01 1",
                    },
                )
        if robot_name == ROBOT_GOOGLE:
            # Phase-A container constraints are execution-only and inactive
            # at reset.  The physical articulation executor fills the live
            # relative transform and enables one weld only after bilateral
            # finger/handle contact has been confirmed.  Perception continues
            # to use the independent direct-actuation API.
            for container_id, moving_body in (
                ("D1", "drawer_D1_tray"),
                ("D2", "drawer_D2_tray"),
                ("C1", "C1_door"),
                ("C2", "C2_door"),
                ("B1", "B1_lid"),
            ):
                ET.SubElement(
                    equality,
                    "weld",
                    {
                        "name": f"google:container_grasp_{container_id}",
                        "body1": gripper_body_name,
                        "body2": moving_body,
                        "active": "false",
                        "solref": "0.01 1",
                    },
                )
    if robot_name == ROBOT_FETCH:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        ET.SubElement(
            contact,
            "exclude",
            {
                "body1": "robot0:torso_lift_link",
                "body2": "robot0:shoulder_lift_link",
            },
        )
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        # Activated only after both fingers contact the box-lid handle. The
        # live relative pose is filled by the physical open action, allowing
        # the arm and the real hinge joint to follow one consistent arc.
        ET.SubElement(
            equality,
            "weld",
            {
                "name": "robot0:open_weld_B1_lid",
                "body1": "robot0:gripper_link",
                "body2": "B1_lid",
                "active": "false",
                "solref": "0.01 1",
            },
        )
        for drawer_name in ("D1", "D2"):
            # A live point constraint preserves the finger-confirmed handle
            # contact while allowing the wrist to yaw naturally during the
            # straight drawer pull.
            ET.SubElement(
                equality,
                "connect",
                {
                    "name": f"robot0:open_connect_{drawer_name}",
                    "body1": "robot0:gripper_link",
                    "body2": f"drawer_{drawer_name}_tray",
                    "anchor": "0 0 0",
                    "active": "false",
                    "solref": "0.01 1",
                },
            )

    return ET.tostring(root, encoding="unicode")


class KitchenScene:
    """
    High-level interface for a kitchen scene experiment.

    Provides:
     - Scene loading and MuJoCo model/data management
     - Container open/close with actuator control
     - Visibility queries (what objects are currently visible)
     - Ground-truth state for evaluation metrics
    """

    def __init__(
        self,
        scene_name: str,
        include_robot: bool = True,
        robot: str | None = None,
        layout_seed: int | None = None,
    ):
        configs = load_all_configs()
        if scene_name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Scene '{scene_name}' not found. Available: {available}")

        self.config = copy.deepcopy(configs[scene_name])
        self.layout_manifest = configure_integrated_target_layout(
            self.config, layout_seed
        )
        self.scene_name = scene_name
        self.robot_name = _selected_robot(include_robot, robot)
        self.has_robot = self.robot_name != ROBOT_NONE
        self.robot_home_qpos = ROBOT_HOME_QPOS.get(self.robot_name, {})
        self.robot_actuators = ROBOT_ACTUATORS.get(self.robot_name, ())

        # Build XML and load model
        print(f"[KitchenScene] Building scene: {scene_name}")
        print(f"  Goal: {self.config.goal}")
        xml_str = build_scene_xml(self.config, robot=self.robot_name)
        model_assets = _load_object_binary_assets()
        if self.robot_name == ROBOT_FETCH:
            model_assets.update(_load_fetch_binary_assets(_fetch_asset_dir()))
        elif self.robot_name == ROBOT_GOOGLE:
            model_assets.update(_load_google_binary_assets(_google_robot_dir()))
        self.model = mujoco.MjModel.from_xml_string(xml_str, assets=model_assets)
        self.data = mujoco.MjData(self.model)
        self._object_instance_records = self._discover_object_instances()

        if self.has_robot:
            self._set_robot_home_pose()

        # Initialize state tracking
        self.state = SceneState()
        self.state.hidden_objects = {
            cid: list(objs) for cid, objs in self.config.container_contents.items()
        }
        self.state.container_open_state = {
            container_id: False for container_id in CONTAINER_JOINTS
        }
        # Countertop objects are always visible
        self.state.visible_objects = set(self.config.countertop_objects.values())
        self.state.visible_object_counts = Counter(self.config.countertop_objects.values())

        # Let all free objects settle onto their support surfaces.
        mujoco.mj_forward(self.model, self.data)
        # Thin scanned utensils need roughly two simulated seconds to finish
        # settling inside the drawer trays without residual jitter.
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
        print(f"  Visible objects: {self.state.visible_objects}")
        print(f"  Hidden objects: {self.state.hidden_objects}")
        print(f"  Required objects: {self.config.required_objects}")
        print(f"  Robot: {self.robot_name}")
        print("  Scene ready.\n")

    def _discover_object_instances(self) -> list[tuple[str, str, str | None]]:
        """Return (instance body, object kind, containing region) records."""
        counts: Counter = Counter()
        records = []

        def add(object_kind: str, region: str | None) -> None:
            counts[object_kind] += 1
            suffix = "" if counts[object_kind] == 1 else f"_{counts[object_kind]}"
            instance_name = f"{object_kind}{suffix}"
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, instance_name
            )
            if body_id >= 0:
                records.append((instance_name, object_kind, region))

        for object_kind in self.config.countertop_objects.values():
            add(object_kind, None)
        for container_id, object_kinds in self.config.container_contents.items():
            for object_kind in object_kinds:
                add(object_kind, container_id)
        return records

    def _set_robot_home_pose(self):
        """Apply deterministic robot joint positions and controller targets."""
        for joint_name, value in self.robot_home_qpos.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise RuntimeError(
                    f"{self.robot_name} joint missing from composed model: "
                    f"{joint_name}"
                )
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = value

        for actuator_name, joint_name, _kp, _lo, _hi in self.robot_actuators:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise RuntimeError(
                    f"{self.robot_name} actuator missing from composed model: "
                    f"{actuator_name}"
                )
            self.data.ctrl[actuator_id] = self.robot_home_qpos[joint_name]

        mujoco.mj_forward(self.model, self.data)

    def get_robot_joint_positions(self) -> dict[str, float]:
        """Return the current selected robot's controlled joint positions."""
        if not self.has_robot:
            return {}
        positions = {}
        for joint_name in self.robot_home_qpos:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            positions[joint_name] = float(
                self.data.qpos[self.model.jnt_qposadr[joint_id]]
            )
        return positions

    def set_robot_joint_targets(self, targets: dict[str, float], steps: int = 500):
        """Command named robot position actuators and advance the simulation."""
        if not self.has_robot:
            raise RuntimeError("This scene was created without a robot")
        actuator_by_joint = {
            joint: (name, lo, hi)
            for name, joint, _kp, lo, hi in self.robot_actuators
        }
        for joint_name, target in targets.items():
            if joint_name not in actuator_by_joint:
                raise ValueError(
                    f"Unknown or unactuated {self.robot_name} joint: {joint_name}"
                )
            actuator_name, lower, upper = actuator_by_joint[joint_name]
            if not lower <= target <= upper:
                raise ValueError(
                    f"Target {target} for {joint_name} outside [{lower}, {upper}]"
                )
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            self.data.ctrl[actuator_id] = target

        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def set_mobile_base_target(
        self, forward: float, lateral: float, yaw: float, steps: int = 750
    ):
        """Command the selected robot's planar base from its initial pose."""
        if not self.has_robot:
            raise RuntimeError("This scene was created without a robot")
        forward_joint, lateral_joint, yaw_joint = ROBOT_BASE_JOINTS[
            self.robot_name
        ]
        self.set_robot_joint_targets(
            {
                forward_joint: forward,
                lateral_joint: lateral,
                yaw_joint: yaw,
            },
            steps=steps,
        )

    def get_visible_objects(self) -> set:
        """Return the set of objects currently visible to the agent."""
        return set(self.state.visible_objects)

    def get_visible_object_instances(self) -> list[tuple[str, str]]:
        """Return actual visible MuJoCo body names paired with semantic kinds."""
        return [
            (instance_name, object_kind)
            for instance_name, object_kind, container_id in self._object_instance_records
            if container_id is None or container_id in self.state.opened_containers
        ]

    def get_instance_source_region(self, instance_name: str) -> str | None:
        """Return the controller-known source after an instance is observable."""
        for known_name, _object_kind, container_id in self._object_instance_records:
            if known_name == instance_name:
                if container_id is None:
                    return "countertop"
                if container_id in self.state.opened_containers:
                    return container_id
                return None
        return None

    def get_region_observation_states(self) -> dict[str, dict]:
        """Return opening/inspection state without exposing hidden contents."""
        states = {}
        for container_id, cinfo in CONTAINER_JOINTS.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, cinfo["joint"]
            )
            joint_position = float(
                self.data.qpos[self.model.jnt_qposadr[joint_id]]
            )
            physical_open = joint_position >= 0.80 * float(cinfo["open_val"])
            states[container_id] = {
                "region_id": container_id,
                "open": physical_open,
                "inspected": container_id in self.state.opened_containers,
                "joint_position": joint_position,
                "open_target": float(cinfo["open_val"]),
                "open_fraction": joint_position / float(cinfo["open_val"]),
            }
        return states

    def get_missing_objects(self) -> list:
        """Return missing required instances, preserving duplicate quantities."""
        required = Counter(self.config.required_objects)
        return list((required - self.state.visible_object_counts).elements())

    def get_searchable_regions(self) -> list:
        """Return list of container IDs that haven't been inspected yet."""
        return [cid for cid in CONTAINER_SLOTS if cid not in self.state.opened_containers]

    def open_container(self, container_id: str, steps: int = 1000) -> list:
        """
        Open a container and return the list of newly visible objects.

        This directly sets the container actuator target and then advances the
        scene controller. It does not imply a robot or manipulation trajectory.
        """
        if container_id not in CONTAINER_JOINTS:
            raise ValueError(f"Unknown container: {container_id}")

        if self.state.container_open_state[container_id]:
            print(f"  [INFO] {container_id} already open.")
            return []

        cinfo = CONTAINER_JOINTS[container_id]
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, cinfo["actuator"]
        )
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, cinfo["joint"]
        )

        # Set actuator target to open position
        self.data.ctrl[actuator_id] = cinfo["open_val"]

        # Step simulation to let it open
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

        joint_position = float(
            self.data.qpos[self.model.jnt_qposadr[joint_id]]
        )
        if joint_position < 0.80 * float(cinfo["open_val"]):
            raise RuntimeError(
                f"{container_id} failed to reach its physical open target: "
                f"joint={joint_position:.4f}, target={cinfo['open_val']:.4f}"
            )

        return self.record_container_opened(container_id)

    def release_storage_fixture(self, container_id: str) -> bool:
        """Release a scene-construction fixture before visual inspection."""
        equality_prefix = STORAGE_FIXTURE_EQUALITIES.get(container_id)
        if equality_prefix is None:
            return False
        released = False
        for equality_id in range(self.model.neq):
            equality_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id
            )
            if equality_name and equality_name.startswith(equality_prefix):
                self.data.eq_active[equality_id] = 0
                released = True
        if not released:
            return False
        mujoco.mj_forward(self.model, self.data)
        return True

    def release_storage_fixture_for_inspection(self, container_id: str) -> bool:
        """Release transport fixtures, but retain C2's upright display weld."""
        if container_id == "C2":
            return False
        return self.release_storage_fixture(container_id)

    def record_container_opened(self, container_id: str) -> list:
        """Update visibility after an API- or motion-driven physical opening."""
        if container_id not in CONTAINER_JOINTS:
            raise ValueError(f"Unknown container: {container_id}")
        self.state.container_open_state[container_id] = True
        if container_id in self.state.opened_containers:
            return []
        self.state.opened_containers.add(container_id)
        newly_visible = self.state.hidden_objects.get(container_id, [])
        for obj in newly_visible:
            self.state.visible_objects.add(obj)
            self.state.visible_object_counts[obj] += 1
            self.state.found_in[obj] = container_id
        print(f"  [OPENED] {container_id} → Found: {newly_visible}")
        return newly_visible

    def set_all_containers_open_snapshot(self) -> None:
        """Load a deterministic all-open perception snapshot without settling.

        C2 and B1 overlap along their full dynamic opening arcs in this compact
        scene. Setting joint coordinates directly is intentional for the
        geometry benchmark: it exposes every region in one reproducible frame
        without asking the contact solver to choose which mechanism may open.
        """
        for container_id, cinfo in CONTAINER_JOINTS.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, cinfo["joint"]
            )
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, cinfo["actuator"]
            )
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = cinfo["open_val"]
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
            self.data.ctrl[actuator_id] = cinfo["open_val"]
            self.record_container_opened(container_id)
        mujoco.mj_forward(self.model, self.data)

    def close_container(self, container_id: str, steps: int = 1000):
        """Close a previously opened container."""
        if container_id not in CONTAINER_JOINTS:
            raise ValueError(f"Unknown container: {container_id}")

        cinfo = CONTAINER_JOINTS[container_id]
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, cinfo["actuator"]
        )
        self.data.ctrl[actuator_id] = 0.0

        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.state.container_open_state[container_id] = False

    def is_task_resolvable(self) -> bool:
        """
        Check if all required objects (or acceptable substitutes) are now visible.
        """
        available = self.state.visible_object_counts.copy()
        required = Counter(self.config.required_objects)

        # Reserve exact matches first so one object cannot satisfy two needs.
        unresolved = []
        for req, count in required.items():
            exact = min(count, available[req])
            available[req] -= exact
            unresolved.extend([req] * (count - exact))

        for req in unresolved:
            for substitute in self.config.substitution_map.get(req, []):
                if available[substitute] > 0:
                    available[substitute] -= 1
                    break
            else:
                return False
        return True

    def get_search_stats(self) -> dict:
        """Return search efficiency statistics."""
        n_opened = len(self.state.opened_containers)
        optimal = self.config.optimal_inspections
        n_useful = sum(
            1 for cid in self.state.opened_containers
            if self.state.hidden_objects.get(cid, [])
        )
        n_redundant = n_opened - n_useful
        n_revisits = 0  # Not possible in this API (containers tracked), but could be

        return {
            "inspections": n_opened,
            "optimal_inspections": optimal,
            "search_efficiency_ratio": optimal / n_opened if n_opened > 0 else 1.0,
            "useful_inspections": n_useful,
            "redundant_inspections": n_redundant,
            "revisits": n_revisits,
            "task_resolvable": self.is_task_resolvable(),
        }

    def launch_viewer(
        self,
        camera: str = FREE_CAMERA,
        actions_panel: bool = True,
        calibration_mode: bool = False,
        task_requirements: str | Path | None = None,
    ):
        """Launch the MuJoCo viewer and, by default, its Actions panel."""
        if camera not in VIEW_CAMERA_CHOICES:
            raise ValueError(
                f"Unknown camera '{camera}'. Choose from: "
                f"{', '.join(VIEW_CAMERA_CHOICES)}"
            )
        if camera in ROBOT_CAMERAS and not self.has_robot:
            raise ValueError(f"Camera '{camera}' requires a robot")
        print(f"[KitchenScene] Launching viewer for: {self.scene_name}")
        print(f"  Starting from camera: {camera}")
        print(
            "  Use the viewer camera menu to switch among: "
            f"{', '.join(VIEW_CAMERA_CHOICES)}"
        )
        print("  Close viewer window to return to script.\n")
        if self.has_robot and actions_panel:
            from mujoco_scenes.mobile_motion import launch_action_viewer
            launch_action_viewer(
                self,
                camera,
                calibration_mode=calibration_mode,
                task_requirements=task_requirements,
            )
            return
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            if camera == FREE_CAMERA:
                mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            else:
                cam_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
                )
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = cam_id
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    def render_frame(self, camera: str = "front_camera",
                     width: int = 1280, height: int = 720) -> np.ndarray:
        """Render a single frame from the given camera."""
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        mujoco.mj_forward(self.model, self.data)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        renderer.update_scene(self.data, camera=cam_id)
        return renderer.render()

    def print_scene_summary(self):
        """Print a formatted summary of the current scene state."""
        print("=" * 60)
        print(f"Scene: {self.scene_name}")
        print(f"Goal:  {self.config.goal}")
        print("-" * 60)
        print(f"Visible objects:     {sorted(self.state.visible_objects)}")
        print(f"Missing (required):  {self.get_missing_objects()}")
        print(f"Containers opened:   {sorted(self.state.opened_containers)}")
        print(f"Regions remaining:   {self.get_searchable_regions()}")
        print(f"Task resolvable:     {self.is_task_resolvable()}")
        stats = self.get_search_stats()
        print(f"Search stats:        {stats}")
        print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────
# CLI / Demo
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MuJoCo kitchen/living-room scene loader"
    )
    parser.add_argument(
        "--environment",
        choices=("auto", "kitchen", "living-room"),
        default="auto",
        help=(
            "Environment family. 'auto' selects living-room for "
            "--scene L1_living_room or L2_living_room_* and kitchen otherwise"
        ),
    )
    parser.add_argument(
        "--scene", type=str, default="S1_coffee_missing_mug",
        help=(
            "Kitchen scene name, L1_living_room, or an L2_living_room_* scene"
        ),
    )
    parser.add_argument(
        "--layout-seed", type=int, default=None,
        help=(
            "For the integrated primary scene, deterministically randomize "
            "three target vessels across C2, B1 and C1"
        ),
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Launch interactive MuJoCo viewer"
    )
    parser.add_argument(
        "--no-actions-panel", action="store_true",
        help="Launch the viewer without the companion Actions panel"
    )
    parser.add_argument(
        "--calibration-mode", action="store_true",
        help=(
            "Show and enable provisional Google Robot pick candidates in the "
            "Actions panel; failures remain guarded and diagnostic"
        ),
    )
    parser.add_argument(
        "--no-robot", action="store_true",
        help="Load the scene without a robot for virtual inspection"
    )
    parser.add_argument(
        "--robot", choices=ROBOT_CHOICES, default=ROBOT_GOOGLE,
        help="Robot backend to compose into the kitchen"
    )
    parser.add_argument(
        "--camera", default=FREE_CAMERA,
        help="Environment camera used for rendering or viewer startup"
    )
    parser.add_argument(
        "--open-container", action="append",
        help=(
            "Open a kitchen container or living-room drawer before output; "
            "may be repeated"
        )
    )
    parser.add_argument(
        "--open-all", action="store_true",
        help="Start with both cupboards, both drawers, and the box open"
    )
    parser.add_argument(
        "--inspect-sequence", nargs="*", metavar="REGION",
        help=(
            "Observe the closed scene, then directly open, virtually face, "
            "and save one evidence stage per region. Requires --no-robot. "
            "With no regions, use D1 D2 C2 B1 C1"
        ),
    )
    parser.add_argument(
        "--task-requirements", type=str, default=None, metavar="YAML",
        help=(
            "Declarative geometry or joint task-role requirements; defaults to "
            "configs/serve_two_person_breakfast.yaml"
        ),
    )
    parser.add_argument(
        "--semantic-detector",
        choices=("none", "yolo_world"),
        default="none",
        help="RGB semantic detector backend (default: none)",
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        metavar="CHECKPOINT",
        help="Pretrained detector checkpoint; defaults to semantic config",
    )
    parser.add_argument(
        "--semantic-config",
        default=None,
        metavar="YAML",
        help="Semantic detection, association, and fusion configuration",
    )
    parser.add_argument(
        "--semantic-vocabulary",
        default=None,
        metavar="YAML",
        help="Independent open-vocabulary detector label configuration",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("auto", "joint", "geometry-only", "semantic-only"),
        default="auto",
        help=(
            "Grounding logic: auto selects production joint grounding for "
            "joint-role tasks and geometry-only for legacy geometry tasks; "
            "geometry-only/semantic-only are diagnostic ablations"
        ),
    )
    parser.add_argument(
        "--pairing-strategy",
        choices=("semantic-role-scoped", "exhaustive-all-pairs"),
        default=None,
        help=(
            "Binary geometry scope: production semantic role gating or the "
            "exhaustive all-pairs timing ablation; defaults to task config"
        ),
    )
    parser.add_argument(
        "--semantic-confidence-threshold",
        type=float,
        default=None,
        help="Detector confidence gate override",
    )
    parser.add_argument(
        "--semantic-min-supporting-views",
        type=int,
        default=None,
        help="Multi-view semantic support count override",
    )
    parser.add_argument(
        "--save-semantic-overlays",
        action="store_true",
        help="Save detector boxes, mask boundaries, and associations",
    )
    parser.add_argument(
        "--stop-on-complete", action="store_true",
        help=(
            "During --inspect-sequence, stop immediately after a COMPLETE "
            "task witness"
        ),
    )
    parser.add_argument(
        "--runs-root", type=str, default="runs",
        help="Root directory for persistent observed-state runs"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Optional deterministic output name for --inspect-sequence"
    )
    parser.add_argument(
        "--point-cloud", type=str, default=None, metavar="OUTPUT_DIR",
        help="Run five-view RGB-D fusion and write per-object PLY clouds"
    )
    parser.add_argument(
        "--point-cloud-width", type=int, default=640,
        help="Width of each point-cloud RGB-D observation (default: 640)"
    )
    parser.add_argument(
        "--point-cloud-height", type=int, default=480,
        help="Height of each point-cloud RGB-D observation (default: 480)"
    )
    parser.add_argument(
        "--point-cloud-voxel", type=float, default=0.003,
        help="World-frame fusion voxel size in metres; 0 disables downsampling"
    )
    parser.add_argument(
        "--render", type=str, default=None,
        help="Render a frame to the specified output path (e.g., frame.png)"
    )
    parser.add_argument(
        "--demo-search", action="store_true",
        help="Run a demo search sequence: open containers one by one"
    )
    parser.add_argument(
        "--list-scenes", action="store_true",
        help="List all available scene configurations"
    )
    args = parser.parse_args()
    if args.inspect_sequence is not None and (
        args.open_all or args.open_container
    ):
        parser.error(
            "--inspect-sequence requires the closed reset state; do not combine "
            "it with --open-all or --open-container"
        )
    if args.stop_on_complete and args.inspect_sequence is None:
        parser.error("--stop-on-complete requires --inspect-sequence")
    if args.inspect_sequence is not None and not args.no_robot:
        parser.error("--inspect-sequence requires --no-robot")

    if args.calibration_mode and not args.viewer:
        parser.error("--calibration-mode requires --viewer")
    if args.calibration_mode and (
        args.robot != ROBOT_GOOGLE or args.no_robot
    ):
        parser.error("--calibration-mode currently requires --robot google")
    if args.calibration_mode and args.no_actions_panel:
        parser.error("--calibration-mode requires the Actions panel")

    if args.list_scenes:
        configs = load_all_configs()
        print("Available scenes:")
        print("-" * 70)
        for name, cfg in configs.items():
            missing = [o for o in cfg.required_objects
                       if o not in set(cfg.countertop_objects.values())]
            print(f"  {name}")
            print(f"    Goal: {cfg.goal[:80]}...")
            print(f"    Missing items: {missing}")
            print(f"    Optimal inspections: {cfg.optimal_inspections}")
            print()
        print("  L1_living_room")
        print("    Goal: Navigate, organize rigid objects, and dust the TV")
        print("    Inspection regions: LEFT_DRAWER, RIGHT_DRAWER")
        from mujoco_scenes.living_room_region_scene import L2_SCENES
        for scene_name in L2_SCENES:
            print(f"  {scene_name}")
            print("    Goal: Ground a refreshment-tray destination region")
        exit(0)

    is_l2_region_scene = args.scene.startswith(
        "L2_living_room_region_ablation1_"
    )
    environment = args.environment
    if environment == "auto":
        environment = (
            "living-room"
            if args.scene == "L1_living_room" or is_l2_region_scene
            else "kitchen"
        )
    if (
        environment == "living-room"
        and args.scene not in {"S1_coffee_missing_mug", "L1_living_room"}
        and not is_l2_region_scene
    ):
        parser.error(
            "--environment living-room requires L1_living_room or an L2 scene"
        )
    if environment == "kitchen" and (
        args.scene == "L1_living_room" or is_l2_region_scene
    ):
        parser.error("Living-room scenes require --environment living-room or auto")

    selected_robot = ROBOT_NONE if args.no_robot else args.robot
    if environment == "living-room":
        if is_l2_region_scene:
            from mujoco_scenes.living_room_region_scene import (
                L2LivingRoomRegionScene,
            )
            scene = L2LivingRoomRegionScene(
                args.scene, robot=selected_robot
            )
        else:
            from mujoco_scenes.living_room_scene import LivingRoomScene
            scene = LivingRoomScene(robot=selected_robot)
    else:
        scene = KitchenScene(
            args.scene, robot=selected_robot, layout_seed=args.layout_seed
        )
    scene.print_scene_summary()

    if args.inspect_sequence is not None:
        if is_l2_region_scene:
            parser.error(
                "Use python -m mujoco_scenes.run_l2_region_ablation for "
                "L2 candidate-region inspection"
            )
        from mujoco_scenes.sequential_inspection import (
            run_sequential_inspection,
        )

        run_sequential_inspection(
            scene,
            args.inspect_sequence
            or getattr(scene, "default_inspection_order", None),
            runs_root=args.runs_root,
            run_id=args.run_id,
            width=args.point_cloud_width,
            height=args.point_cloud_height,
            voxel_size=args.point_cloud_voxel,
            task_requirements=args.task_requirements,
            stop_on_complete=args.stop_on_complete,
            semantic_backend=args.semantic_detector,
            semantic_model=args.semantic_model,
            semantic_config_path=args.semantic_config,
            semantic_vocabulary_path=args.semantic_vocabulary,
            semantic_confidence_threshold=(
                args.semantic_confidence_threshold
            ),
            semantic_min_supporting_views=(
                args.semantic_min_supporting_views
            ),
            grounding_mode=args.grounding_mode,
            pairing_strategy=args.pairing_strategy,
            save_semantic_overlays=args.save_semantic_overlays,
        )

    containers_to_open = list(args.open_container or [])
    if args.open_all:
        scene.set_all_containers_open_snapshot()
    else:
        for container in containers_to_open:
            scene.open_container(container)

    if args.point_cloud:
        from mujoco_scenes.geometry_checker import GeometryChecker, print_run_summary

        checker = GeometryChecker(
            scene,
            width=args.point_cloud_width,
            height=args.point_cloud_height,
            voxel_size=args.point_cloud_voxel,
        )
        point_cloud_run = checker.run(args.point_cloud)
        print_run_summary(point_cloud_run)

    if args.demo_search:
        if environment != "kitchen":
            parser.error("--demo-search is currently a kitchen-only workflow")
        print("\n[DEMO] Running sequential container search...\n")
        missing = scene.get_missing_objects()
        print(f"  Missing objects to find: {missing}\n")

        search_order = scene.config.optimal_search_order
        for container in search_order:
            print(f"  → Inspecting {container}...")
            found = scene.open_container(container)
            print(f"    Newly visible: {found}")

            # Check remaining
            still_missing = scene.get_missing_objects()
            print(f"    Still missing: {still_missing}")

            if scene.is_task_resolvable():
                print("\n  ✓ All required objects (or substitutes) found!")
                break
            print()

        scene.print_scene_summary()

    if args.render:
        print(f"\n[RENDER] Saving frame to {args.render}")
        frame = scene.render_frame(camera=args.camera)
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(args.render)
        print(f"  Saved {frame.shape[1]}x{frame.shape[0]} frame.")

    if args.viewer:
        if is_l2_region_scene:
            scene.launch_viewer(camera=args.camera)
            exit(0)
        viewer_options = {
            "camera": args.camera,
            "actions_panel": not args.no_actions_panel,
            "calibration_mode": args.calibration_mode,
        }
        if environment == "kitchen":
            viewer_options["task_requirements"] = args.task_requirements
        scene.launch_viewer(**viewer_options)
