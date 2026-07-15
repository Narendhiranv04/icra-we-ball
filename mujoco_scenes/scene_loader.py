"""
Kitchen Scene Loader for 'Reasoning Before Planning' MuJoCo Experiments.

Dynamically loads the kitchen_base.xml, injects objects into containers
and onto the countertop based on a YAML scene config, and provides
an API for opening/closing containers and querying visibility.
"""

import yaml
import copy
import time
import xml.etree.ElementTree as ET
from collections import Counter
from importlib.util import find_spec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

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


# ── Fetch mobile manipulator ─────────────────────────────────────────────
# The visual/collision meshes and kinematic tree come from the maintained
# Gymnasium-Robotics Fetch assets. We adapt its benchmark-fixed base to three
# controllable planar joints and use position actuators for the arm/gripper.
FETCH_PACKAGE = "gymnasium_robotics"
FETCH_ASSET_SUBDIR = Path("envs") / "assets" / "fetch"
FETCH_BASE_POSE = {
    "pos": "0 -1.05 0",
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
    # Navigation/search pose: arm tucked to the side so the head camera has
    # an unobstructed view of the work surface and closed containers.
    "robot0:shoulder_pan_joint": 1.15,
    "robot0:shoulder_lift_joint": 0.75,
    "robot0:upperarm_roll_joint": -0.20,
    "robot0:elbow_flex_joint": -1.35,
    "robot0:forearm_roll_joint": 0.0,
    "robot0:wrist_flex_joint": -0.50,
    "robot0:wrist_roll_joint": 0.0,
    "robot0:r_gripper_finger_joint": 0.035,
    "robot0:l_gripper_finger_joint": 0.035,
}

FETCH_ACTUATORS = (
    # name, joint, kp, ctrl_min, ctrl_max
    ("robot0:base_forward_actuator", "robot0:base_forward_joint", 6000, -1.0, 1.0),
    ("robot0:base_lateral_actuator", "robot0:base_lateral_joint", 6000, -1.0, 1.0),
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


# ── physically supported object placement ────────────────────────────────
# Slot Z values identify the top of a support surface, not the object centre.
# The loader adds the object's lowest-point offset so objects start just above
# shelves/trays instead of intersecting them.
OBJECT_SUPPORT_HEIGHT = {
    "mug": 0.04065, "cup": 0.03076, "glass": 0.055,
    "plate": 0.01336, "small_plate": 0.00735, "bowl": 0.02750,
    "spoon": 0.01045, "fork": 0.00773, "knife": 0.00762,
    "stirrer": 0.003, "spatula": 0.006, "tongs": 0.005,
    "kettle": 0.06817, "coffee_jar": 0.09287, "sugar_jar": 0.06724,
    "coffee_can": 0.07009, "sugar_box": 0.08802,
    "milk_carton": 0.060, "tea_box": 0.01944, "bread": 0.025,
    "butter": 0.015, "jam_jar": 0.040, "napkin": 0.003,
    "biscuits": 0.020, "pot_with_soup": 0.050,
    "gso_canister_distractor": 0.07187,
    "gso_spatula_distractor": 0.01234,
}

UTENSIL_OBJECTS = {
    "spoon", "fork", "knife", "stirrer", "spatula", "tongs",
    "gso_spatula_distractor",
}
CENTRED_DRAWER_OBJECTS = {
    "spoon", "fork", "knife", "stirrer", "tongs",
    "gso_spatula_distractor",
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
            (-0.09, 0.070, -0.052),
            (-0.09, 0.015, -0.052),
            (-0.09, -0.040, -0.052),
            (-0.09, -0.080, -0.052),
        ],
    },
    "D2": {
        "parent_body": "drawer_D2_tray",
        "slots": [
            (-0.09, 0.070, -0.052),
            (-0.09, 0.015, -0.052),
            (-0.09, -0.040, -0.052),
            (-0.09, -0.080, -0.052),
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
    "counter_spot_1": (-0.35, -0.08, 0.770),
    "counter_spot_2": (-0.15, -0.08, 0.770),
    "counter_spot_3": (0.05, -0.08, 0.770),
    "counter_spot_4": (0.20, -0.08, 0.770),
    "counter_spot_5": (-0.25, -0.22, 0.770),
    "counter_spot_6": (0.0, -0.22, 0.770),
}

CAMERAS = (
    "left_shoulder_camera",
    "right_shoulder_camera",
    "overhead_camera",
    "wrist_camera",
    "front_camera",
)
ROBOT_CAMERAS = ("head_camera_rgb",)
CAMERA_CHOICES = CAMERAS + ROBOT_CAMERAS

# Joint/actuator names for each container
CONTAINER_JOINTS = {
    "C1": {"joint": "C1_door_joint", "actuator": "C1_door_actuator", "open_val": 1.4},
    "C2": {"joint": "C2_door_joint", "actuator": "C2_door_actuator", "open_val": 1.4},
    "D1": {"joint": "D1_slide_joint", "actuator": "D1_slide_actuator", "open_val": 0.24},
    "D2": {"joint": "D2_slide_joint", "actuator": "D2_slide_actuator", "open_val": 0.24},
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


@dataclass
class SceneState:
    """Runtime state tracking visibility and search progress."""
    opened_containers: set = field(default_factory=set)
    visible_objects: set = field(default_factory=set)       # Objects currently visible
    visible_object_counts: Counter = field(default_factory=Counter)
    hidden_objects: dict = field(default_factory=dict)      # {container: [objects]} - ground truth
    found_in: dict = field(default_factory=dict)            # {object: container_found_in}
    object_positions: dict = field(default_factory=dict)    # {object: (x, y, z)}


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
        ("robot0:base_lateral_joint", "slide", "0 1 0", "-1 1", "500"),
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


def _load_object_binary_assets() -> dict[str, bytes]:
    """Load prepared kitchen OBJ and texture files for in-memory MJCF compile."""
    supported = {".obj", ".png", ".jpg", ".jpeg"}
    assets = {}
    for path in OBJECT_MESHES_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            key = path.relative_to(OBJECT_MESHES_DIR).as_posix()
            assets[key] = path.read_bytes()
    return assets


def build_scene_xml(config: SceneConfig, include_robot: bool = True) -> str:
    """
    Take kitchen_base.xml, inject objects from the scene config, and return
    the complete XML string ready for mujoco.MjModel.from_xml_string().
    """
    tree = ET.parse(KITCHEN_BASE)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    obj_lib, object_assets = _parse_object_library()
    asset_root = root.find("asset")
    for element in object_assets:
        asset_root.append(copy.deepcopy(element))

    if include_robot:
        _inject_fetch_robot(root, _fetch_asset_dir())

    # Track unique instance counters for duplicates
    instance_count = {}

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
        if obj_name not in obj_lib:
            print(f"  [WARNING] Object '{obj_name}' not found in object library, skipping.")
            return

        instance_name = _get_instance_name(obj_name)
        obj_elem = copy.deepcopy(obj_lib[obj_name])

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
        if instance_name != obj_name:
            old_name = obj_name
            new_name = instance_name
            obj_elem.set("name", new_name)
            for child in obj_elem.iter():
                for attr in ["name", "joint"]:
                    val = child.get(attr)
                    if val and old_name in val:
                        child.set(attr, val.replace(old_name, new_name))

        worldbody.append(obj_elem)

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
            pos = COUNTER_SPOTS[spot]
            _inject_object(obj_name, pos)
        else:
            print(f"  [WARNING] Unknown counter spot '{spot}', skipping {obj_name}.")

    # ── Place container objects ───────────────────────────────────────────
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
        if container_id == "C1":
            lower_slots = [slots[0], slots[3]]
            shelf_slots = [slots[1], slots[2]]
            allocated_slots = []
            for obj_name in objects:
                is_tall = 2 * OBJECT_SUPPORT_HEIGHT.get(obj_name, 0.03) > 0.15
                preferred = shelf_slots if is_tall else lower_slots
                fallback = lower_slots if is_tall else shelf_slots
                allocated_slots.append((preferred or fallback).pop(0))
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
            # Centred scanned utensils use the drawer middle lane; legacy +X
            # primitives retain the left lane. Compact objects use the right
            # lane so realistic-width napkins/boxes cannot overlap utensils.
            if container_id in {"D1", "D2"} and obj_name not in UTENSIL_OBJECTS:
                slot_rel_pos[0] = 0.08
            elif container_id in {"D1", "D2"} and obj_name in CENTRED_DRAWER_OBJECTS:
                slot_rel_pos[0] = 0.0
            world_pos = parent_body_pos + np.array(slot_rel_pos)
            if container_id == "B1" and obj_name == "coffee_jar":
                # The coffee jar remains slightly taller than B1's closed
                # interior. Store it centred on its side along the longer X
                # dimension. The shorter sugar jar now fits upright in the
                # enlarged box and is deliberately left upright so it cannot
                # roll indefinitely on its cylindrical collision proxy.
                world_pos[0] = parent_body_pos[0]
                horizontal_radius = 0.03962
                _inject_object(
                    obj_name,
                    world_pos,
                    quat="0.7071068 0 0.7071068 0",
                    support_height_override=horizontal_radius,
                )
            else:
                _inject_object(obj_name, world_pos)

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

    def __init__(self, scene_name: str, include_robot: bool = True):
        configs = load_all_configs()
        if scene_name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Scene '{scene_name}' not found. Available: {available}")

        self.config = configs[scene_name]
        self.scene_name = scene_name
        self.has_robot = include_robot

        # Build XML and load model
        print(f"[KitchenScene] Building scene: {scene_name}")
        print(f"  Goal: {self.config.goal}")
        xml_str = build_scene_xml(self.config, include_robot=include_robot)
        model_assets = _load_object_binary_assets()
        if include_robot:
            model_assets.update(_load_fetch_binary_assets(_fetch_asset_dir()))
        self.model = mujoco.MjModel.from_xml_string(xml_str, assets=model_assets)
        self.data = mujoco.MjData(self.model)

        if include_robot:
            self._set_fetch_home_pose()

        # Initialize state tracking
        self.state = SceneState()
        self.state.hidden_objects = {
            cid: list(objs) for cid, objs in self.config.container_contents.items()
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
        print(f"  Fetch robot: {'enabled' if self.has_robot else 'disabled'}")
        print(f"  Scene ready.\n")

    def _set_fetch_home_pose(self):
        """Apply deterministic Fetch joint positions and controller targets."""
        for joint_name, value in FETCH_HOME_QPOS.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise RuntimeError(f"Fetch joint missing from composed model: {joint_name}")
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = value

        for actuator_name, joint_name, _kp, _lo, _hi in FETCH_ACTUATORS:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            self.data.ctrl[actuator_id] = FETCH_HOME_QPOS[joint_name]

        mujoco.mj_forward(self.model, self.data)

    def get_robot_joint_positions(self) -> dict[str, float]:
        """Return the current Fetch base, torso, arm, head and gripper joints."""
        if not self.has_robot:
            return {}
        positions = {}
        for joint_name in FETCH_HOME_QPOS:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            positions[joint_name] = float(
                self.data.qpos[self.model.jnt_qposadr[joint_id]]
            )
        return positions

    def set_robot_joint_targets(self, targets: dict[str, float], steps: int = 500):
        """Command named Fetch position actuators and advance the simulation."""
        if not self.has_robot:
            raise RuntimeError("This scene was created without the Fetch robot")
        actuator_by_joint = {
            joint: (name, lo, hi)
            for name, joint, _kp, lo, hi in FETCH_ACTUATORS
        }
        for joint_name, target in targets.items():
            if joint_name not in actuator_by_joint:
                raise ValueError(f"Unknown or unactuated Fetch joint: {joint_name}")
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
        """Command the Fetch planar base relative to its initial floor pose."""
        self.set_robot_joint_targets(
            {
                "robot0:base_forward_joint": forward,
                "robot0:base_lateral_joint": lateral,
                "robot0:base_yaw_joint": yaw,
            },
            steps=steps,
        )

    def get_visible_objects(self) -> set:
        """Return the set of objects currently visible to the agent."""
        return set(self.state.visible_objects)

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

        This simulates the robot opening the container (setting actuator target)
        and then perceiving what's inside.
        """
        if container_id not in CONTAINER_JOINTS:
            raise ValueError(f"Unknown container: {container_id}")

        if container_id in self.state.opened_containers:
            print(f"  [INFO] {container_id} already open.")
            return []

        cinfo = CONTAINER_JOINTS[container_id]
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, cinfo["actuator"]
        )

        # Set actuator target to open position
        self.data.ctrl[actuator_id] = cinfo["open_val"]

        # Step simulation to let it open
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

        # Mark as opened and reveal contents
        self.state.opened_containers.add(container_id)
        newly_visible = self.state.hidden_objects.get(container_id, [])

        for obj in newly_visible:
            self.state.visible_objects.add(obj)
            self.state.visible_object_counts[obj] += 1
            self.state.found_in[obj] = container_id

        print(f"  [OPENED] {container_id} → Found: {newly_visible}")
        return newly_visible

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

    def launch_viewer(self, camera: str = "front_camera"):
        """Launch an interactive, continuously stepping MuJoCo viewer."""
        if camera not in CAMERA_CHOICES:
            raise ValueError(
                f"Unknown camera '{camera}'. Choose from: {', '.join(CAMERA_CHOICES)}"
            )
        if camera in ROBOT_CAMERAS and not self.has_robot:
            raise ValueError(f"Camera '{camera}' requires the Fetch robot")
        print(f"[KitchenScene] Launching viewer for: {self.scene_name}")
        print(f"  Starting from camera: {camera}")
        print(f"  Use the viewer camera menu to switch among: {', '.join(CAMERA_CHOICES)}")
        print(f"  Close viewer window to return to script.\n")
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
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

    parser = argparse.ArgumentParser(description="Kitchen Scene Loader")
    parser.add_argument(
        "--scene", type=str, default="S1_coffee_missing_mug",
        help="Scene name from scene_configs.yaml"
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Launch interactive MuJoCo viewer"
    )
    parser.add_argument(
        "--no-robot", action="store_true",
        help="Load only the kitchen (useful for environment-only debugging)"
    )
    parser.add_argument(
        "--camera", choices=CAMERA_CHOICES, default="front_camera",
        help="Fixed camera used for rendering or as the viewer's starting view"
    )
    parser.add_argument(
        "--open-container", action="append", choices=CONTAINER_JOINTS,
        help="Open a container before rendering/viewing; may be repeated"
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
        exit(0)

    # Load scene
    scene = KitchenScene(args.scene, include_robot=not args.no_robot)
    scene.print_scene_summary()

    for container in args.open_container or []:
        scene.open_container(container)

    if args.demo_search:
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
                print(f"\n  ✓ All required objects (or substitutes) found!")
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
        scene.launch_viewer(camera=args.camera)
