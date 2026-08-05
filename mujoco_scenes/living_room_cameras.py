"""Camera-rig constants shared by the living-room scene and perception."""

from __future__ import annotations

import math


# Google Robot has a wheeled base rather than articulated feet. These two
# cameras sit on its lower front corners and look beneath low furniture.
FOOT_CAMERA_MOUNTS = (
    ("left_foot_camera", 0.16, -8.0),
    ("right_foot_camera", -0.16, 8.0),
)
SOFA_CAMERAS = tuple(name for name, _lateral, _yaw in FOOT_CAMERA_MOUNTS)

# Five overlapping outward views form a full-surround rig above the fixed head
# shell. They remain attached to the mobile base instead of moving with the arm.
TOP_CAMERA_MOUNTS = (
    ("top_front_camera", 0.0),
    ("top_front_left_camera", 72.0),
    ("top_rear_left_camera", 144.0),
    ("top_rear_right_camera", -144.0),
    ("top_front_right_camera", -72.0),
)
TOP_CAMERAS = tuple(name for name, _yaw in TOP_CAMERA_MOUNTS)
ROBOT_DEBUG_CAMERAS = SOFA_CAMERAS + TOP_CAMERAS


def camera_xyaxes(yaw_degrees: float, pitch_degrees: float) -> str:
    """Return MuJoCo camera axes for a yawed, downward-pitched view."""
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    right = (math.sin(yaw), -math.cos(yaw), 0.0)
    up = (
        math.sin(pitch) * math.cos(yaw),
        math.sin(pitch) * math.sin(yaw),
        math.cos(pitch),
    )
    return " ".join(f"{value:.6f}" for value in (*right, *up))


__all__ = [
    "FOOT_CAMERA_MOUNTS",
    "ROBOT_DEBUG_CAMERAS",
    "SOFA_CAMERAS",
    "TOP_CAMERA_MOUNTS",
    "TOP_CAMERAS",
    "camera_xyaxes",
]
