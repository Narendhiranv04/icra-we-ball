"""Controlled L2 living-room scenes for functional support-region grounding."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
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
L2_BASE = ROOT / "assets" / "living_room_region_base.xml"
L2_ABLATION1_SCENES = (
    "L2_living_room_region_ablation1_primary",
    "L2_living_room_region_ablation1_initial_complete",
    "L2_living_room_region_ablation1_exhaustion",
)
L2_ABLATION2_SCENES = (
    "L2_living_room_region_ablation2_primary",
    "L2_living_room_region_ablation2_drinks_dedicated",
    "L2_living_room_region_ablation2_controls_shared",
    "L2_living_room_region_ablation2_exhaustion",
    "L2_living_room_region_ablation2_permuted",
)
L2_SCENES = L2_ABLATION1_SCENES + L2_ABLATION2_SCENES
L2_ABLATION2_BASE = (
    ROOT / "assets" / "living_room_region_ablation2_base.xml"
)
L2_CAMERAS = (
    "l2_camera_left",
    "l2_camera_right",
    "l2_camera_top",
    "l2_camera_front",
    "l2_camera_close",
)
L2_GOAL = (
    "Place the refreshment tray on a suitable living-room surface within "
    "easy reach of the sofa."
)
L2_ABLATION2_GOAL = (
    "Set up the living room for two people watching television. Place one "
    "drink in a separate accessible region beside each seating position, "
    "and keep the TV remote and game controller together in one shared "
    "accessible region."
)


def build_l2_region_xml(
    scene_name: str,
    robot: str = ROBOT_GOOGLE,
) -> str:
    """Compose one controlled L2 variant with Google Robot or no robot."""
    if scene_name not in L2_SCENES:
        raise ValueError(f"Unknown L2 region scene: {scene_name}")
    if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
        raise ValueError("L2 region scenes support robot google or none")
    ablation2 = scene_name in L2_ABLATION2_SCENES
    root = ET.parse(L2_ABLATION2_BASE if ablation2 else L2_BASE).getroot()
    if scene_name in L2_ABLATION1_SCENES and scene_name.endswith("_exhaustion"):
        # Retain recognizable coffee-table context while making its observed
        # support patch robustly too small for the tray. Runtime inference
        # never reads this construction-time value.
        top = next(
            geom
            for geom in root.iter("geom")
            if geom.get("name") == "l2_coffee_table_top"
        )
        top.set("size", "0.155 0.100 0.04")
        # Separate the deliberately undersized table from the sofa arm.  At
        # high RGB-D resolution a same-height strip of the adjacent vertical
        # arm face can otherwise join the stage-local support evidence and
        # inflate the PCA footprint.  This is a scene-design separation only;
        # runtime measurement still receives no intended size or pose.
        exhaustion_center_x = 0.62
        top_position = np.fromstring(top.get("pos", ""), sep=" ")
        top_position[0] = exhaustion_center_x
        top.set(
            "pos", " ".join(f"{value:.5f}" for value in top_position)
        )
        for geom in root.iter("geom"):
            name = geom.get("name", "")
            if name.startswith("l2_coffee_leg_"):
                position = np.fromstring(geom.get("pos", ""), sep=" ")
                position[0] = (
                    exhaustion_center_x
                    + np.sign(position[0] - 0.42) * 0.10
                )
                position[1] = 0.20 + np.sign(position[1] - 0.20) * 0.06
                geom.set("pos", " ".join(f"{value:.5f}" for value in position))
    if ablation2 and scene_name.endswith("_exhaustion"):
        # Keep all candidates visible and plausible, but make the only
        # control-semantic surface too narrow for simultaneous two-object
        # packing. Runtime geometry still measures this from RGB-D.
        top = next(
            geom
            for geom in root.iter("geom")
            if geom.get("name") == "a2_control_table_top"
        )
        top.set("size", "0.16 0.065 0.040")
    if ablation2 and scene_name.endswith("_permuted"):
        # Change the visible layout and free-instance creation order without
        # making the detector solve a different appearance/occlusion problem.
        # The two equivalent drink payloads exchange positions; the controls
        # retain the primary scene's reliably recognized viewpoints. Runtime
        # association must still recover fresh generic IDs from image evidence.
        first_drink = root.find(".//body[@name='a2_drink_left']")
        second_drink = root.find(".//body[@name='a2_drink_right']")
        first_position = first_drink.get("pos")
        first_drink.set("pos", second_drink.get("pos"))
        second_drink.set("pos", first_position)
        worldbody = root.find("worldbody")
        payload_bodies = [
            child
            for child in list(worldbody)
            if child.tag == "body"
            and child.get("name", "").startswith("a2_")
        ]
        for child in payload_bodies:
            worldbody.remove(child)
        for child in reversed(payload_bodies):
            worldbody.append(child)
    if robot == ROBOT_GOOGLE:
        _inject_google_robot(root, _google_robot_dir())
    return ET.tostring(root, encoding="unicode")


class L2LivingRoomRegionScene:
    """Compiled L2 room plus only the interfaces needed by observation."""

    goal = L2_GOAL
    point_cloud_cameras = L2_CAMERAS
    payload_instance_name = "l2_refreshment_tray"

    def __init__(
        self,
        scene_name: str = L2_SCENES[0],
        robot: str = ROBOT_GOOGLE,
    ):
        if scene_name not in L2_SCENES:
            raise ValueError(f"Unknown L2 region scene: {scene_name}")
        if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
            raise ValueError("L2 region scenes support google or none")
        self.scene_name = scene_name
        self.goal = (
            L2_ABLATION2_GOAL
            if scene_name in L2_ABLATION2_SCENES
            else L2_GOAL
        )
        self.robot_name = robot
        self.has_robot = robot == ROBOT_GOOGLE
        print(f"[L2RegionScene] Building scene: {scene_name}")
        print(f"  Goal: {self.goal}")
        assets = (
            _load_google_binary_assets(_google_robot_dir())
            if self.has_robot
            else {}
        )
        self.model = mujoco.MjModel.from_xml_string(
            build_l2_region_xml(scene_name, robot),
            assets=assets,
        )
        self.data = mujoco.MjData(self.model)
        self._set_robot_home_pose()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(600):
            mujoco.mj_step(self.model, self.data)
        print(f"  Robot: {robot}")
        print(
            "  Candidate supports: "
            f"{5 if scene_name in L2_ABLATION2_SCENES else 3}"
        )
        print("  Scene ready.\n")

    def _set_robot_home_pose(self) -> None:
        if not self.has_robot:
            return
        for joint_name, value in GOOGLE_HOME_QPOS.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise RuntimeError(f"Google Robot joint missing: {joint_name}")
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        for actuator_name, joint_name, _kp, _lower, _upper in GOOGLE_ACTUATORS:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise RuntimeError(
                    f"Google Robot actuator missing: {actuator_name}"
                )
            self.data.ctrl[actuator_id] = GOOGLE_HOME_QPOS[joint_name]

    def get_visible_object_instances(self) -> list[tuple[str, str]]:
        """Expose only the fixed payload as an object-level observation."""
        if self.scene_name in L2_ABLATION2_SCENES:
            # Ablation 2 discovers its four generic payload IDs from visible
            # segmentation instances and RGB semantics in one initial capture.
            # Do not leak simulator body names through the legacy object API.
            return []
        return [(self.payload_instance_name, "refreshment_tray")]

    def render_frame(
        self,
        camera: str = "l2_camera_front",
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        if camera not in L2_CAMERAS:
            raise ValueError(f"Unknown L2 camera: {camera}")
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            mujoco.mj_forward(self.model, self.data)
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()

    def print_scene_summary(self) -> None:
        print("=" * 60)
        print(f"Scene: {self.scene_name}")
        print(f"Goal:  {self.goal}")
        print("-" * 60)
        if self.scene_name in L2_ABLATION2_SCENES:
            print("Candidate regions:  5, all visible initially")
            print("Fixed payloads:      4, discovered from RGB-D evidence")
            print("Seating targets:     2, spatially distinct")
        else:
            print(
                "Candidate regions:  "
                "RUG_PATCH, SMALL_SIDE_TABLE, COFFEE_TABLE"
            )
            print("Fixed payload:       one observed refreshment tray")
        print(f"Robot:               {self.robot_name}")
        print("=" * 60)

    def launch_viewer(self, camera: str = FREE_CAMERA) -> None:
        if camera != FREE_CAMERA and camera not in L2_CAMERAS:
            raise ValueError(f"Unknown L2 camera: {camera}")
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
