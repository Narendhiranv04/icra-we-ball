"""Compact workshop scene for joint region and object alternatives."""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

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
WORKSHOP_INSPECTION_RIG_CONFIG = (
    ROOT / "configs" / "workshop_inspection_rigs.yaml"
)
WORKSHOP_REGIONS = ("LEFT_DRAWER", "RIGHT_DRAWER")
WORKSHOP_CAMERAS = (
    "workshop_camera_left",
    "workshop_camera_right",
    "workshop_camera_top",
    "workshop_camera_front",
    "workshop_camera_close",
)
INITIAL_OBJECTS = (
    ("frame_joint", "frame_joint"),
    ("workshop_hammer", "hammer"),
    ("workshop_large_nail", "nail"),
    ("workshop_marker", "marker"),
)
REGION_OBJECTS = {
    "LEFT_DRAWER": (
        ("workshop_flat_driver", "screwdriver"),
        ("workshop_short_screw", "screw"),
    ),
    "RIGHT_DRAWER": (
        ("workshop_phillips_driver", "screwdriver"),
        ("workshop_medium_screw", "screw"),
    ),
}


@dataclass
class WorkshopObservationState:
    opened_containers: set[str] = field(default_factory=set)
    container_open_state: dict[str, bool] = field(
        default_factory=lambda: {
            region_id: False for region_id in WORKSHOP_REGIONS
        }
    )


def build_workshop_xml(robot: str = ROBOT_GOOGLE) -> str:
    if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
        raise ValueError("Workshop supports robot google or none")
    root = ET.parse(WORKSHOP_BASE).getroot()
    if robot == ROBOT_GOOGLE:
        _inject_google_robot(root, _google_robot_dir())
    return ET.tostring(root, encoding="unicode")


class WorkshopScene:
    """Workshop geometry plus the generic closed-region observation API."""

    scene_name = "W1_workshop_joint_alternatives"
    goal = "Fasten the two frame pieces using a compatible observed tool system."
    point_cloud_cameras = WORKSHOP_CAMERAS
    inspection_rig_config_path = WORKSHOP_INSPECTION_RIG_CONFIG
    initial_observation_region = "workbench"
    default_inspection_order = WORKSHOP_REGIONS
    inspection_interference: dict[str, str] = {}

    def __init__(self, robot: str = ROBOT_GOOGLE):
        if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
            raise ValueError("Workshop supports robot google or none")
        self.robot_name = robot
        self.has_robot = robot == ROBOT_GOOGLE
        self.state = WorkshopObservationState()
        assets = (
            _load_google_binary_assets(_google_robot_dir())
            if self.has_robot
            else {}
        )
        self.model = mujoco.MjModel.from_xml_string(
            build_workshop_xml(robot), assets=assets
        )
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
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
            for actuator_name, joint_name, *_rest in GOOGLE_ACTUATORS:
                actuator_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                self.data.ctrl[actuator_id] = GOOGLE_HOME_QPOS[joint_name]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def get_visible_object_instances(self) -> list[tuple[str, str]]:
        visible = list(INITIAL_OBJECTS)
        for region_id in self.state.opened_containers:
            visible.extend(REGION_OBJECTS[region_id])
        return visible

    def get_instance_source_region(self, instance_name: str) -> str | None:
        if any(name == instance_name for name, _kind in INITIAL_OBJECTS):
            return "workbench"
        for region_id, objects in REGION_OBJECTS.items():
            if any(name == instance_name for name, _kind in objects):
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

    def _drawer_actuator_id(self, region_id: str) -> int:
        actuator_name = {
            "LEFT_DRAWER": "left_tool_drawer_actuator",
            "RIGHT_DRAWER": "right_fastener_drawer_actuator",
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
        actuator_id = self._drawer_actuator_id(region_id)
        self.data.ctrl[actuator_id] = 0.72
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.state.container_open_state[region_id] = True
        was_new = region_id not in self.state.opened_containers
        self.state.opened_containers.add(region_id)
        return (
            [name for name, _kind in REGION_OBJECTS[region_id]]
            if was_new
            else []
        )

    def close_container(self, region_id: str, steps: int = 300) -> None:
        actuator_id = self._drawer_actuator_id(region_id)
        self.data.ctrl[actuator_id] = 0.0
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.state.container_open_state[region_id] = False

    def print_scene_summary(self) -> None:
        print(f"Scene: {self.scene_name}")
        print(f"Goal:  {self.goal}")
        print(f"Robot: {self.robot_name}")
        print(f"Inspected regions: {sorted(self.state.opened_containers)}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=(ROBOT_GOOGLE, ROBOT_NONE), default=ROBOT_GOOGLE)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--camera", default=FREE_CAMERA)
    parser.add_argument("--open", choices=WORKSHOP_REGIONS, action="append", default=[])
    arguments = parser.parse_args()
    scene = WorkshopScene(arguments.robot)
    for region_id in arguments.open:
        scene.open_container(region_id)
    scene.print_scene_summary()
    if arguments.viewer:
        scene.launch_viewer(arguments.camera)


if __name__ == "__main__":
    main()
