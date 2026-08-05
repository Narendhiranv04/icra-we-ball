"""Scene-specific rigid-object manipulation for the living room."""

from __future__ import annotations

import math
from dataclasses import replace

import mujoco
import numpy as np

from mujoco_scenes.generic_manipulation import (
    CalibratedPickPlaceExecutor,
    SimplePickSpec,
)
from mujoco_scenes.robot_profiles import manipulation_profile


_controller_yaw = -0.20
CONTROLLER_TOP_DOWN_ROTATION = np.array(
    (
        (math.cos(_controller_yaw), -math.sin(_controller_yaw), 0.0),
        (math.sin(_controller_yaw), math.cos(_controller_yaw), 0.0),
        (0.0, 0.0, 1.0),
    )
) @ manipulation_profile("google").top_down_rotation


LIVING_ROOM_PICK_SPECS = {
    "remote_control": SimplePickSpec(
        "TV remote",
        "remote_control_grasp",
        0.012,
        grasp_z_offset=0.020,
        required_contact_geoms=("remote_control_collision",),
    ),
    "living_room_mug": SimplePickSpec(
        "Ceramic mug",
        "living_room_mug_grasp",
        0.055,
        grasp_z_offset=0.011,
        required_contact_geoms=("living_room_mug_collision",),
    ),
    "hardback_book": SimplePickSpec(
        "Hardback book",
        "hardback_book_grasp",
        0.018,
        grasp_z_offset=0.014,
        required_contact_geoms=("hardback_book_collision",),
        final_tracking_tolerance=0.055,
        carry_grip_relaxation=0.040,
    ),
    "game_controller": SimplePickSpec(
        "Game controller",
        "game_controller_grasp",
        0.022,
        grasp_z_offset=0.015,
        required_contact_geoms=(
            "game_controller_core",
            "game_controller_left_grip",
            "game_controller_right_grip",
        ),
        top_down_rotation=CONTROLLER_TOP_DOWN_ROTATION,
        final_tracking_tolerance=0.020,
    ),
    "rigid_duster": SimplePickSpec(
        "Rigid TV duster",
        "rigid_duster_grasp",
        0.025,
        grasp_z_offset=0.016,
        required_contact_geoms=("rigid_duster_handle_collision",),
        final_tracking_tolerance=0.020,
    ),
}

TABLETOP_OBJECTS = (
    "remote_control",
    "living_room_mug",
    "hardback_book",
    "game_controller",
)
CALIBRATED_LIVING_ROOM_OBJECTS: tuple[str, ...] = (
    "remote_control",
    "living_room_mug",
    "hardback_book",
    "game_controller",
    "rigid_duster",
)
PLACE_SITE_BY_OBJECT = {
    "remote_control": "table_place_remote",
    "living_room_mug": "coaster_right_mug_place",
    "hardback_book": "table_place_book",
    "game_controller": "table_place_controller",
    "rigid_duster": "duster_return_site",
}

INITIAL_OBJECT_LOCATIONS = {
    "remote_control": "home",
    "living_room_mug": "home",
    "hardback_book": "home",
    "game_controller": "home",
    "rigid_duster": "duster",
}

LIVING_ROOM_ARM_COMMAND_SPEED = 1.55
LIVING_ROOM_INTERMEDIATE_TRACKING_TOLERANCE = 0.085


class LivingRoomManipulationExecutor:
    """Select the correct live base stance, then delegate guarded pick/place."""

    def __init__(self, scene, calibration_mode: bool = False):
        self.scene = scene
        self.calibration_mode = calibration_mode
        self.executor: CalibratedPickPlaceExecutor | None = None
        self.status = "Living-room manipulation idle"
        self._last_failure: str | None = None
        self.object_locations = dict(INITIAL_OBJECT_LOCATIONS)
        self.pick_source_location: str | None = None
        self.pending_place_location: str | None = None
        self.pending_place_object: str | None = None

    @property
    def all_pick_specs(self):
        return LIVING_ROOM_PICK_SPECS

    @property
    def calibrated_objects(self) -> frozenset[str]:
        return frozenset(CALIBRATED_LIVING_ROOM_OBJECTS)

    @property
    def busy(self) -> bool:
        return self.executor is not None and self.executor.busy

    @property
    def held_object(self) -> str | None:
        return None if self.executor is None else self.executor.held_object

    @property
    def failure(self) -> str | None:
        if self.executor is not None and self.executor.failure:
            return self.executor.failure
        return self._last_failure

    @property
    def mode(self) -> str:
        return "idle" if self.executor is None else self.executor.mode

    @property
    def navigation_safe(self) -> bool:
        return self.executor is None or self.executor.navigation_safe

    @property
    def can_place(self) -> bool:
        return self.executor is not None and self.executor.can_place

    def _base_qpos(self) -> np.ndarray:
        names = (
            "google:base_forward_joint",
            "google:base_lateral_joint",
            "google:base_yaw_joint",
        )
        addresses = [
            self.scene.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
            ]
            for name in names
        ]
        return self.scene.data.qpos[addresses].copy()

    def required_pick_location(self, object_name: str) -> str:
        if object_name not in self.object_locations:
            raise ValueError(f"Unknown living-room object: {object_name}")
        return self.object_locations[object_name]

    def _placement_target(self) -> tuple[str, str]:
        if self.held_object is None:
            raise RuntimeError("No held object has a placement target")
        source = self.pick_source_location
        if self.held_object == "hardback_book":
            return (
                ("bookshelf", "media_shelf_book_place")
                if source == "home"
                else ("home", "table_place_book")
            )
        if self.held_object == "game_controller":
            return (
                ("drawer", "drawer_place_controller")
                if source == "home"
                else ("home", "table_place_controller")
            )
        if self.held_object == "rigid_duster":
            return "duster", "duster_return_site"
        return "home", PLACE_SITE_BY_OBJECT[self.held_object]

    @property
    def place_destination(self) -> str | None:
        if self.held_object is None:
            return None
        return self._placement_target()[0]

    @property
    def place_label(self) -> str:
        if self.held_object == "hardback_book":
            return (
                "Place book on shelf"
                if self.pick_source_location == "home"
                else "Return book to table"
            )
        if self.held_object == "game_controller":
            return (
                "Store controller in drawer"
                if self.pick_source_location == "home"
                else "Return controller to table"
            )
        if self.held_object == "rigid_duster":
            return "Return duster"
        if self.held_object == "living_room_mug":
            return "Place mug on coaster"
        return "Place held object"

    @staticmethod
    def _approach_distance(location: str) -> float:
        return {
            "duster": 0.0,
            "bookshelf": 0.10,
            "drawer": 0.08,
        }.get(location, 0.15)

    def _specs_for_stance(
        self,
        object_name: str,
        base_stance: np.ndarray,
        current_location: str,
    ) -> dict[str, SimplePickSpec]:
        specs = dict(LIVING_ROOM_PICK_SPECS)
        if object_name in TABLETOP_OBJECTS and current_location == "home":
            table_x, table_y, table_yaw = self.scene.table_pose
            cosine = math.cos(table_yaw)
            sine = math.sin(table_yaw)
            # Preserve the reset calibration in the moving table's frame.
            # At reset this is the original world carry target
            # (0.0, -0.78, 0.94), 43 cm south of the table centre.
            carry_position = np.array(
                (
                    table_x + 0.43 * sine,
                    table_y - 0.43 * cosine,
                    0.94,
                )
            )
            table_rotation = np.array(
                (
                    (cosine, -sine, 0.0),
                    (sine, cosine, 0.0),
                    (0.0, 0.0, 1.0),
                )
            )
            object_rotation = specs[object_name].top_down_rotation
            top_down = table_rotation @ (
                manipulation_profile("google").top_down_rotation
                if object_rotation is None
                else object_rotation
            )
            specs[object_name] = replace(
                specs[object_name],
                carry_position=carry_position,
                top_down_rotation=top_down,
            )
        else:
            world_x = -float(base_stance[1])
            world_y = -1.25 + float(base_stance[0])
            yaw = float(base_stance[2])
            specs[object_name] = replace(
                specs[object_name],
                carry_position=np.array(
                    (
                        world_x - 0.47 * math.sin(yaw),
                        world_y + 0.47 * math.cos(yaw),
                        0.94,
                    )
                ),
            )
        return specs

    def request_pick(self, object_name: str, current_location: str) -> None:
        if self.busy or self.held_object is not None:
            raise RuntimeError("The gripper is not available for another pick")
        if object_name not in LIVING_ROOM_PICK_SPECS:
            raise ValueError(f"Unknown living-room pick object: {object_name}")
        required_location = self.required_pick_location(object_name)
        if current_location != required_location:
            raise RuntimeError(
                f"Pick {object_name} requires Move ({required_location}) first"
            )
        base_stance = self._base_qpos()
        specs = self._specs_for_stance(
            object_name, base_stance, current_location
        )
        approach_distance = self._approach_distance(current_location)
        table_yaw = float(self.scene.table_pose[2])
        # The composed mobile joints translate in world Y / negative world X.
        # Express the short approach in the table frame so the calibration
        # remains explicit even though the current living-room table is fixed.
        approach_delta = np.array(
            (
                approach_distance * math.cos(table_yaw),
                approach_distance * math.sin(table_yaw),
                0.0,
            )
        )
        self.executor = CalibratedPickPlaceExecutor(
            self.scene.model,
            self.scene.data,
            "google",
            scene_name=None,
            calibration_mode=self.calibration_mode,
            pick_specs_override=specs,
            calibrated_objects_override=CALIBRATED_LIVING_ROOM_OBJECTS,
            base_stance=base_stance,
            base_approach_delta=approach_delta,
            arm_command_speed=LIVING_ROOM_ARM_COMMAND_SPEED,
            intermediate_tracking_tolerance=(
                LIVING_ROOM_INTERMEDIATE_TRACKING_TOLERANCE
            ),
        )
        self._last_failure = None
        self.executor.request_pick(object_name)
        self.pick_source_location = current_location
        self.status = self.executor.status

    def request_place(self, current_location: str) -> None:
        if self.executor is None or self.held_object is None:
            raise RuntimeError("Pick a living-room object before placing")
        required_location, place_site = self._placement_target()
        if current_location != required_location:
            raise RuntimeError(
                f"Place {self.held_object} requires Move ({required_location}) first"
            )
        base_stance = self._base_qpos()
        approach_distance = self._approach_distance(current_location)
        self.executor.base_stance = base_stance
        self.executor.base_manipulation_target = base_stance + np.array(
            (approach_distance, 0.0, 0.0)
        )
        self.pending_place_location = current_location
        self.pending_place_object = self.held_object
        self.executor.request_place(place_site)
        self.status = self.executor.status

    def update(self) -> None:
        if self.executor is None:
            return
        placed_object = self.pending_place_object
        self.executor.update()
        self.status = self.executor.status
        if self.executor.failure:
            self._last_failure = self.executor.failure
        if (
            placed_object is not None
            and self.pending_place_location is not None
            and self.executor.mode == "idle"
            and self.executor.held_object is None
        ):
            self.object_locations[placed_object] = self.pending_place_location
            self.pending_place_location = None
            self.pending_place_object = None
            self.pick_source_location = None

    def progress(self) -> float:
        return 0.0 if self.executor is None else self.executor.progress()
