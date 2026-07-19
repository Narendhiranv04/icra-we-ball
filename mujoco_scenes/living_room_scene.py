"""Rigid living-room environment for Google Robot interaction experiments."""

from __future__ import annotations

import argparse
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
LIVING_ROOM_BASE = ROOT / "assets" / "living_room_base.xml"
LIVING_ROOM_CAMERAS = (
    "overhead_camera",
    "room_corner_camera",
    "tv_camera",
    "table_camera",
    "wrist_camera",
    "head_camera_rgb",
)
LIVING_ROOM_VIEW_CAMERAS = (FREE_CAMERA,) + LIVING_ROOM_CAMERAS
LIVING_ROOM_OBJECTS = (
    "remote_control",
    "living_room_mug",
    "hardback_book",
    "coaster_left",
    "coaster_right",
    "game_controller",
    "rigid_duster",
)
PICKABLE_OBJECTS = (
    "remote_control",
    "living_room_mug",
    "hardback_book",
    "game_controller",
    "rigid_duster",
)
TV_CELL_COUNT = 15
DUST_FADE_RESPONSE = 5.0
LIVING_ROOM_FORWARD_LIMITS = (-1.0, 2.10)


def build_living_room_xml(robot: str = ROBOT_GOOGLE) -> str:
    """Compose the rigid living room with Google Robot or no robot."""
    if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
        raise ValueError("Living room currently supports --robot google or none")
    root = ET.parse(LIVING_ROOM_BASE).getroot()
    if robot == ROBOT_GOOGLE:
        _inject_google_robot(root, _google_robot_dir())
        forward_joint = next(
            joint
            for joint in root.iter("joint")
            if joint.get("name") == "google:base_forward_joint"
        )
        forward_joint.set("range", "-1 2.10")
        forward_actuator = next(
            item
            for item in root.iter("position")
            if item.get("name") == "google:base_forward_actuator"
        )
        forward_actuator.set("ctrlrange", "-1 2.10")
        equality = root.find("equality")
        assert equality is not None
        for object_name in PICKABLE_OBJECTS:
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


class LivingRoomScene:
    """Compiled living room plus resettable interaction state."""

    scene_name = "L1_living_room"
    goal = (
        "Navigate the living room, organize rigid tabletop objects, and "
        "dust the flat-screen TV"
    )

    def __init__(self, robot: str = ROBOT_GOOGLE):
        if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
            raise ValueError("Living room currently supports google or none")
        self.robot_name = robot
        self.has_robot = robot == ROBOT_GOOGLE
        print("[LivingRoomScene] Building rigid living room")
        print(f"  Goal: {self.goal}")
        xml = build_living_room_xml(robot)
        assets = (
            _load_google_binary_assets(_google_robot_dir())
            if self.has_robot
            else {}
        )
        self.model = mujoco.MjModel.from_xml_string(xml, assets=assets)
        self.data = mujoco.MjData(self.model)
        self.cleaned_cells: set[int] = set()
        self.tv_power_on = False
        self._initial_geom_rgba = self.model.geom_rgba.copy()
        self._initial_mat_rgba = self.model.mat_rgba.copy()
        self._initial_mat_emission = self.model.mat_emission.copy()
        self._initial_eq_data = self.model.eq_data.copy()
        self._initial_eq_solref = self.model.eq_solref.copy()
        self._dust_film_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "dust_screen_film"
        )
        self._tv_power_led_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "tv_power_led"
        )
        if self._dust_film_geom_id < 0 or self._tv_power_led_geom_id < 0:
            raise RuntimeError("Living-room TV visual interface is incomplete")
        self._dust_initial_alpha = float(
            self.model.geom_rgba[self._dust_film_geom_id, 3]
        )
        self._dust_target_alpha = self._dust_initial_alpha
        self.reset()
        print(f"  Robot: {self.robot_name}")
        print(f"  Rigid scene objects: {', '.join(LIVING_ROOM_OBJECTS)}")
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
                raise RuntimeError(f"Google Robot actuator missing: {actuator_name}")
            self.data.ctrl[actuator_id] = GOOGLE_HOME_QPOS[joint_name]

    def reset(self, settle_steps: int = 1000) -> None:
        """Restore robot, table, clutter, constraints, and visible dust."""
        mujoco.mj_resetData(self.model, self.data)
        self.model.geom_rgba[:] = self._initial_geom_rgba
        self.model.mat_rgba[:] = self._initial_mat_rgba
        self.model.mat_emission[:] = self._initial_mat_emission
        self.model.eq_data[:] = self._initial_eq_data
        self.model.eq_solref[:] = self._initial_eq_solref
        self.data.eq_active[:] = self.model.eq_active0
        self.cleaned_cells.clear()
        self._dust_target_alpha = self._dust_initial_alpha
        self.tv_power_on = False
        self._set_robot_home_pose()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise ValueError(f"Unknown living-room body: {name}")
        return body_id

    def site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, name
        )
        if site_id < 0:
            raise ValueError(f"Unknown living-room site: {name}")
        return site_id

    @property
    def table_pose(self) -> tuple[float, float, float]:
        body_id = self.body_id("coffee_table")
        rotation = self.data.xmat[body_id].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        position = self.data.xpos[body_id]
        return float(position[0]), float(position[1]), yaw

    @property
    def dust_coverage(self) -> float:
        return len(self.cleaned_cells) / TV_CELL_COUNT

    @property
    def dust_opacity(self) -> float:
        return float(self.model.geom_rgba[self._dust_film_geom_id, 3])

    def mark_tv_cell_clean(self, index: int) -> None:
        if not 0 <= index < TV_CELL_COUNT:
            raise ValueError(f"TV cell must be in [0, {TV_CELL_COUNT - 1}]")
        self.cleaned_cells.add(index)
        self._dust_target_alpha = self._dust_initial_alpha * (
            1.0 - self.dust_coverage
        )

    def update_visual_effects(self, steps: int = 1) -> None:
        """Ease the visible dust film toward verified coverage state."""
        if steps < 1:
            raise ValueError("Visual-effect steps must be positive")
        current = self.dust_opacity
        elapsed = float(self.model.opt.timestep) * steps
        blend = 1.0 - float(np.exp(-DUST_FADE_RESPONSE * elapsed))
        updated = current + (self._dust_target_alpha - current) * blend
        if abs(updated - self._dust_target_alpha) < 1e-5:
            updated = self._dust_target_alpha
        self.model.geom_rgba[self._dust_film_geom_id, 3] = updated

    def set_tv_power(self, enabled: bool) -> None:
        """Update the visible TV screen state without changing collision."""
        material_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "tv_screen"
        )
        if material_id < 0:
            raise RuntimeError("Missing TV screen material")
        self.tv_power_on = bool(enabled)
        if self.tv_power_on:
            self.model.mat_rgba[material_id] = (0.10, 0.34, 0.62, 1.0)
            self.model.mat_emission[material_id] = 0.50
            self.model.geom_rgba[self._tv_power_led_geom_id] = (
                0.05,
                0.85,
                0.16,
                1.0,
            )
        else:
            self.model.mat_rgba[material_id] = self._initial_mat_rgba[material_id]
            self.model.mat_emission[material_id] = self._initial_mat_emission[
                material_id
            ]
            self.model.geom_rgba[self._tv_power_led_geom_id] = (
                self._initial_geom_rgba[self._tv_power_led_geom_id]
            )

    def render_frame(
        self,
        camera: str = "room_corner_camera",
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        if camera not in LIVING_ROOM_CAMERAS:
            raise ValueError(f"Choose a fixed camera from: {LIVING_ROOM_CAMERAS}")
        if not self.has_robot and camera in {"wrist_camera", "head_camera_rgb"}:
            raise ValueError(f"Camera {camera} requires Google Robot")
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        mujoco.mj_forward(self.model, self.data)
        renderer.update_scene(self.data, camera=camera)
        frame = renderer.render().copy()
        renderer.close()
        return frame

    def launch_viewer(
        self,
        camera: str = FREE_CAMERA,
        actions_panel: bool = True,
        calibration_mode: bool = False,
    ) -> None:
        if camera not in LIVING_ROOM_VIEW_CAMERAS:
            raise ValueError(f"Unknown living-room camera: {camera}")
        if not self.has_robot and camera in {"wrist_camera", "head_camera_rgb"}:
            raise ValueError(f"Camera {camera} requires Google Robot")
        if self.has_robot and actions_panel:
            from mujoco_scenes.living_room_actions import launch_living_room_actions

            launch_living_room_actions(
                self, camera=camera, calibration_mode=calibration_mode
            )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Robot living room")
    parser.add_argument("--robot", choices=(ROBOT_GOOGLE, ROBOT_NONE), default=ROBOT_GOOGLE)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-actions-panel", action="store_true")
    parser.add_argument("--calibration-mode", action="store_true")
    parser.add_argument("--camera", choices=LIVING_ROOM_VIEW_CAMERAS, default=FREE_CAMERA)
    parser.add_argument("--render", help="Save one rendered PNG frame")
    args = parser.parse_args()
    if args.calibration_mode and (not args.viewer or args.no_actions_panel):
        parser.error("--calibration-mode requires --viewer and the Actions panel")
    scene = LivingRoomScene(robot=args.robot)
    if args.render:
        from PIL import Image

        fixed_camera = (
            "room_corner_camera" if args.camera == FREE_CAMERA else args.camera
        )
        Image.fromarray(scene.render_frame(fixed_camera)).save(args.render)
        print(f"[LivingRoomScene] Saved render to {args.render}")
    if args.viewer:
        scene.launch_viewer(
            camera=args.camera,
            actions_panel=not args.no_actions_panel,
            calibration_mode=args.calibration_mode,
        )


if __name__ == "__main__":
    main()
