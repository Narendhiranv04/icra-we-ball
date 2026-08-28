"""Compact single-arm frame-joint repair scene."""

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
    _apply_robot_home_pose,
    _inject_google_robot,
    _load_google_binary_assets,
    _validate_render_dimensions,
)


ROOT = Path(__file__).resolve().parent
WORKSHOP_BASE = ROOT / "assets" / "workshop_base.xml"
WORKSHOP_INSPECTION_RIG_CONFIG = (
    ROOT / "configs" / "workshop_inspection_rigs.yaml"
)
WORKSHOP_REGIONS = ("LEFT_DRAWER", "TOOL_CABINET")
WORKSHOP_FUNCTIONAL_REGIONS = (
    "FRAME_FIXTURE",
    "SCREW_STAGING_TRAY",
    "TOOL_CABINET",
)
WORKSHOP_CAMERAS = (
    "workshop_camera_left",
    "workshop_camera_right",
    "workshop_camera_top",
    "workshop_camera_front",
    "workshop_camera_close",
)
INITIAL_OBJECTS = (
    ("workshop_frame_joint", "fixture_held_frame_joint"),
    ("workshop_joint_seal", "protective_joint_seal"),
)
REGION_OBJECTS = {
    "LEFT_DRAWER": (
        ("workshop_manual_driver", "phillips_screwdriver"),
        ("workshop_short_screw", "short_screw"),
    ),
    "TOOL_CABINET": (
        ("workshop_power_driver", "powered_screwdriver"),
        ("workshop_long_screw", "long_screw"),
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
    joint_repaired: bool = False
    joint_seal_location: str = "FRAME_JOINT"


def build_workshop_xml(robot: str = ROBOT_GOOGLE) -> str:
    if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
        raise ValueError("Workshop supports robot google or none")
    root = ET.parse(WORKSHOP_BASE).getroot()
    if robot == ROBOT_GOOGLE:
        _inject_google_robot(root, _google_robot_dir())
    return ET.tostring(root, encoding="unicode")


class WorkshopScene:
    """Workshop geometry plus a bounded closed-region observation API.

    The fixture and captive screw guide replace a second robot hand. The
    drawer and tool cabinet begin closed and can be inspected independently.
    This module models only the scene and observable state; it does not
    implement grasping, screw driving, or task-and-motion planning.
    """

    scene_name = "W1_workshop_joint_prerequisites"
    goal = (
        "Remove the protective seal, then repair the fixture-held frame joint "
        "using the first compatible observed driver and screw."
    )
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
            _apply_robot_home_pose(
                self.model,
                self.data,
                robot_name="Google Robot",
                home_qpos=GOOGLE_HOME_QPOS,
                actuators=GOOGLE_ACTUATORS,
            )
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

    def get_task_scene_state(self) -> dict[str, object]:
        """Return non-privileged symbolic state exposed by the scene.

        Container open/closed state is visible and does not reveal contents.
        """
        return {
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
            "tool_cabinet": {
                "open": self.state.container_open_state["TOOL_CABINET"],
            },
        }

    def move_joint_seal_to_tray(self, steps: int = 300) -> None:
        """Apply the ground-truth debug transition for a successful removal.

        Future execution code should replace this teleport with a planned
        grasp-and-place motion and update the same observable state afterward.
        """
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "workshop_joint_seal_free",
        )
        qpos_address = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_address : qpos_address + 3] = (-0.84, 0.19, 0.72)
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        dof_address = self.model.jnt_dofadr[joint_id]
        self.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.state.joint_seal_location = "SCREW_STAGING_TRAY"

    def _container_actuator_id(self, region_id: str) -> int:
        actuator_name = {
            "LEFT_DRAWER": "left_tool_drawer_actuator",
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
        target = 1.25 if region_id == "TOOL_CABINET" else 0.72
        self.data.ctrl[actuator_id] = target
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

    def close_container(self, region_id: str, steps: int = 1000) -> None:
        actuator_id = self._container_actuator_id(region_id)
        self.data.ctrl[actuator_id] = 0.0
        joint_id = int(self.model.actuator_trnid[actuator_id, 0])
        qpos_address = self.model.jnt_qposadr[joint_id]
        dof_address = self.model.jnt_dofadr[joint_id]
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if (
                abs(self.data.qpos[qpos_address]) < 1e-3
                and abs(self.data.qvel[dof_address]) < 1e-3
            ):
                break
        self.state.container_open_state[region_id] = False

    def print_scene_summary(self) -> None:
        print(f"Scene: {self.scene_name}")
        print(f"Goal:  {self.goal}")
        print(f"Robot: {self.robot_name}")
        print(f"Inspected regions: {sorted(self.state.opened_containers)}")
        print(
            "Functional regions: "
            + ", ".join(WORKSHOP_FUNCTIONAL_REGIONS)
        )
        cabinet_state = self.get_task_scene_state()["tool_cabinet"]
        print("Tool cabinet: " + ("open" if cabinet_state["open"] else "closed"))
        print(
            "Joint access: "
            + (
                "covered by protective seal"
                if not self.get_task_scene_state()["joint_access"]["clear"]
                else "clear"
            )
        )

    def render_frame(
        self,
        camera: str = "workshop_camera_front",
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        width, height = _validate_render_dimensions(width, height)
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
    parser.add_argument("--remove-seal", action="store_true")
    arguments = parser.parse_args()
    scene = WorkshopScene(arguments.robot)
    for region_id in arguments.open:
        scene.open_container(region_id)
    if arguments.remove_seal:
        scene.move_joint_seal_to_tray()
    scene.print_scene_summary()
    if arguments.viewer:
        scene.launch_viewer(arguments.camera)


if __name__ == "__main__":
    main()
