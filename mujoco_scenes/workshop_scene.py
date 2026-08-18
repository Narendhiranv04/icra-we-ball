"""Integrated Workshop (W1) Joint Object and Region Function Benchmark Scene."""

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

WORKSHOP_FUNCTIONAL_WORK_SURFACES = (
    "MAIN_WORKBENCH_ZONE",
    "TOOL_CART_TOP",
    "NARROW_WALL_SHELF",
    "HIGH_CABINET_TOP",
)

WORKSHOP_FUNCTIONAL_PARTS_CONTAINERS = (
    "PARTS_TRAY",
    "HARDWARE_BIN",
    "TOOLBOX_COMPARTMENT",
)

WORKSHOP_CAMERAS = (
    "workshop_camera_left",
    "workshop_camera_right",
    "workshop_camera_top",
    "workshop_camera_front",
    "workshop_camera_close",
)

# Canonical object catalog with authored semantic and physical categories
WORKSHOP_OBJECT_CATALOG = {
    "workshop_long_phillips_driver": {
        "kind": "phillips_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.18,
        "tip_profile": "PH2",
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.026 * 0.035,
    },
    "workshop_stubby_phillips_driver": {
        "kind": "phillips_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.035,  # Too short to reach recessed joint
        "tip_profile": "PH2",
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.014 * 0.040,
    },
    "workshop_flathead_screwdriver": {
        "kind": "slotted_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.15,
        "tip_profile": "SLOTTED",  # Incompatible with Phillips recess
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.022 * 0.030,
    },
    "workshop_power_driver": {
        "kind": "powered_screwdriver",
        "functions": ["can_drive_screw"],
        "reach_m": 0.12,
        "tip_profile": "PH2",
        "tip_width_m": 0.006,
        "bounding_area_m2": 0.21 * 0.08,  # Large footprint
    },
    "workshop_medium_phillips_screw": {
        "kind": "medium_screw",
        "functions": ["can_fasten"],
        "length_m": 0.045,
        "head_diameter_m": 0.016,
        "shaft_diameter_m": 0.007,
        "recess_profile": "PH2",
        "recess_width_m": 0.0065,
        "bounding_area_m2": 0.045 * 0.016,
    },
    "workshop_short_phillips_screw": {
        "kind": "short_screw",
        "functions": ["can_fasten"],
        "length_m": 0.018,  # Inadequate joint engagement depth
        "head_diameter_m": 0.014,
        "shaft_diameter_m": 0.006,
        "recess_profile": "PH2",
        "recess_width_m": 0.0065,
        "bounding_area_m2": 0.018 * 0.014,
    },
    "workshop_long_phillips_screw": {
        "kind": "long_screw",
        "functions": ["can_fasten"],
        "length_m": 0.085,  # Too long for standard hole
        "head_diameter_m": 0.018,
        "shaft_diameter_m": 0.009,
        "recess_profile": "PH2",
        "recess_width_m": 0.0065,
        "bounding_area_m2": 0.085 * 0.018,
    },
    "workshop_hex_bolt": {
        "kind": "hex_bolt",
        "functions": ["can_fasten"],
        "length_m": 0.050,
        "head_diameter_m": 0.018,
        "shaft_diameter_m": 0.008,
        "recess_profile": "HEX",  # Incompatible with Phillips driver
        "recess_width_m": 0.010,
        "bounding_area_m2": 0.050 * 0.018,
    },
    "workshop_pliers": {
        "kind": "combination_pliers",
        "functions": ["can_grip"],
        "reach_m": 0.08,
        "bounding_area_m2": 0.19 * 0.05,
    },
    "workshop_combination_wrench": {
        "kind": "combination_wrench",
        "functions": ["can_turn_hex"],
        "reach_m": 0.10,
        "bounding_area_m2": 0.21 * 0.04,
    },
    "workshop_ratchet_wrench": {
        "kind": "ratchet_wrench",
        "functions": ["can_turn_hex"],
        "reach_m": 0.12,
        "bounding_area_m2": 0.24 * 0.045,
    },
    "workshop_wooden_mallet": {
        "kind": "wooden_mallet",
        "functions": ["can_hammer"],
        "reach_m": 0.12,
        "bounding_area_m2": 0.28 * 0.11,
    },
    "workshop_joint_seal": {
        "kind": "protective_joint_seal",
        "functions": ["seal_cover"],
        "bounding_area_m2": 0.06 * 0.05,
    },
    "workshop_frame_joint": {
        "kind": "fixture_held_frame_joint",
        "functions": ["target_workpiece"],
        "hole_diameter_m": 0.007,
        "hole_depth_m": 0.030,
    },
}

INITIAL_OBJECTS = (
    ("workshop_frame_joint", "fixture_held_frame_joint"),
    ("workshop_joint_seal", "protective_joint_seal"),
)

PICKABLE_OBJECTS = tuple(
    name for name in WORKSHOP_OBJECT_CATALOG.keys() if name != "workshop_frame_joint"
)


def _load_workshop_variants_config() -> dict[str, Any]:
    with open(WORKSHOP_VARIANTS_CONFIG, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@dataclass
class WorkshopObservationState:
    opened_containers: set[str] = field(default_factory=set)
    container_open_state: dict[str, bool] = field(
        default_factory=lambda: {region_id: False for region_id in WORKSHOP_REGIONS}
    )
    joint_repaired: bool = False
    joint_seal_location: str = "FRAME_JOINT"


def build_workshop_xml(
    robot: str = ROBOT_GOOGLE, variant: str = "F0_BASE"
) -> str:
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

    root = ET.parse(WORKSHOP_BASE).getroot()

    # Configure objects inside storage regions based on the variant specification
    region_body_map = {
        "LEFT_DRAWER": "left_tool_drawer",
        "RIGHT_DRAWER": "right_tool_drawer",
        "TOOL_CABINET": "tool_cabinet",
    }

    # Gather all potential storage item bodies from base XML
    all_storage_objects = set()
    for obj_list in storage_contents.values():
        all_storage_objects.update(obj_list)

    # Filter bodies in each storage container to match the variant
    for region_id, parent_body_name in region_body_map.items():
        parent_body = None
        for body in root.iter("body"):
            if body.get("name") == parent_body_name:
                parent_body = body
                break
        if parent_body is None:
            continue

        target_objects = set(storage_contents.get(region_id, []))
        children_to_remove = []
        for child in list(parent_body):
            child_name = child.get("name", "")
            if child.tag == "body" and child_name.startswith("workshop_"):
                if child_name not in target_objects:
                    children_to_remove.append(child)

        for child in children_to_remove:
            parent_body.remove(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        # In I2_NO_WORK_SURFACE: Obstruct / remove candidate work surfaces
        if variant == "I2_NO_WORK_SURFACE":
            for body in list(worldbody):
                if body.get("name") in ("workshop_tool_cart", "workshop_narrow_shelf"):
                    worldbody.remove(body)

        # In I3_NO_PARTS_CONTAINER: Remove candidate parts containers
        if variant == "I3_NO_PARTS_CONTAINER":
            for body in list(worldbody):
                if body.get("name") in ("workshop_parts_tray", "workshop_hardware_bin"):
                    worldbody.remove(body)

    if robot == ROBOT_GOOGLE:
        _inject_google_robot(root, _google_robot_dir())
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")

        present_bodies = {b.get("name") for b in root.iter("body")}
        for object_name in PICKABLE_OBJECTS:
            if object_name in present_bodies:
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


class WorkshopScene:
    """Workshop geometry plus a bounded closed-region observation API.

    The fixture and captive screw guide replace a second robot hand.
    The left drawer, right drawer, and tool cabinet begin closed and can be
    inspected independently. This module models the scene, physical settling,
    and observable candidates without leaking privileged ground truth.
    """

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
                kind = WORKSHOP_OBJECT_CATALOG.get(obj_name, {}).get("kind", "tool")
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

    def get_candidate_work_surfaces(self) -> list[dict[str, Any]]:
        """Return candidate work surface proposals for spatial grounding."""
        return [
            {
                "region_id": "MAIN_WORKBENCH_ZONE",
                "center_world_m": [0.0, 0.38, 0.68],
                "dimensions_m": [0.60, 0.33, 0.05],
                "usable_area_m2": 0.60 * 0.33,
            },
            {
                "region_id": "TOOL_CART_TOP",
                "center_world_m": [0.92, 0.32, 0.80],
                "dimensions_m": [0.32, 0.21, 0.05],
                "usable_area_m2": 0.32 * 0.21,
            },
            {
                "region_id": "NARROW_WALL_SHELF",
                "center_world_m": [-0.70, 0.60, 1.05],
                "dimensions_m": [0.24, 0.07, 0.03],
                "usable_area_m2": 0.24 * 0.07,
            },
            {
                "region_id": "HIGH_CABINET_TOP",
                "center_world_m": [0.0, 0.58, 1.10],
                "dimensions_m": [0.22, 0.14, 0.03],
                "usable_area_m2": 0.22 * 0.14,
            },
        ]

    def get_candidate_parts_containers(self) -> list[dict[str, Any]]:
        """Return candidate parts container proposals."""
        return [
            {
                "region_id": "PARTS_TRAY",
                "center_world_m": [-0.44, 0.22, 0.71],
                "dimensions_m": [0.32, 0.22, 0.045],
                "cavity_volume_m3": 0.30 * 0.20 * 0.035,
                "is_open": True,
            },
            {
                "region_id": "HARDWARE_BIN",
                "center_world_m": [0.44, 0.22, 0.71],
                "dimensions_m": [0.16, 0.12, 0.08],
                "cavity_volume_m3": 0.14 * 0.10 * 0.07,
                "is_open": True,
            },
            {
                "region_id": "TOOLBOX_COMPARTMENT",
                "center_world_m": [0.92, 0.32, 0.85],
                "dimensions_m": [0.38, 0.18, 0.14],
                "cavity_volume_m3": 0.36 * 0.16 * 0.12,
                "is_open": True,
            },
        ]

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
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr : qpos_adr + 3] = (-0.44, 0.22, 0.73)
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
        print("Candidate work surfaces: " + ", ".join(WORKSHOP_FUNCTIONAL_WORK_SURFACES))
        print("Candidate parts containers: " + ", ".join(WORKSHOP_FUNCTIONAL_PARTS_CONTAINERS))

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
