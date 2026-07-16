"""Robot-specific interfaces used by generic kitchen motion code.

The scene composer owns model construction.  This module owns the small set of
names and conventions that motion code must know about each robot backend.
Keeping these declarations together makes adding and calibrating another robot
an explicit process instead of scattering string substitutions across planners.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MobileRobotProfile:
    base_joints: tuple[str, str, str]
    base_actuators: tuple[str, str, str]
    body_prefix: str
    home_y: float
    forward_limits: tuple[float, float]


@dataclass(frozen=True)
class ManipulationProfile:
    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    finger_joints: tuple[str, str]
    finger_actuators: tuple[str, str]
    finger_contact_geoms: tuple[frozenset[str], frozenset[str]]
    gripper_body: str
    grip_site: str
    top_down_rotation: np.ndarray
    open_command: float
    closed_command: float
    close_step: float
    carry_position: np.ndarray
    home_seed: np.ndarray
    navigation_joints: np.ndarray
    supported_objects: tuple[str, ...]


MOBILE_PROFILES = {
    "fetch": MobileRobotProfile(
        base_joints=(
            "robot0:base_forward_joint",
            "robot0:base_lateral_joint",
            "robot0:base_yaw_joint",
        ),
        base_actuators=(
            "robot0:base_forward_actuator",
            "robot0:base_lateral_actuator",
            "robot0:base_yaw_actuator",
        ),
        body_prefix="robot0:",
        home_y=-1.10,
        forward_limits=(-1.0, 1.0),
    ),
    "google": MobileRobotProfile(
        base_joints=(
            "google:base_forward_joint",
            "google:base_lateral_joint",
            "google:base_yaw_joint",
        ),
        base_actuators=(
            "google:base_forward_actuator",
            "google:base_lateral_actuator",
            "google:base_yaw_actuator",
        ),
        body_prefix="google:",
        # Leave additional clearance from the serving table for folding the
        # arm, rotating, and transporting an attached object.
        home_y=-1.25,
        forward_limits=(-1.0, 1.25),
    ),
}


GOOGLE_RIGHT_FINGER_GEOMS = frozenset(
    f"google:right_finger_pad_{index}" for index in range(6)
)
GOOGLE_LEFT_FINGER_GEOMS = frozenset(
    f"google:left_finger_pad_{index}" for index in range(6)
)


# Google Robot's gripper site extends along local +Z.  For a top-down grasp,
# local +Z points down, local +Y is the jaw closing axis, and local +X completes
# a right-handed frame.
GOOGLE_TOP_DOWN_ROTATION = np.array(
    ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
)


MANIPULATION_PROFILES = {
    "google": ManipulationProfile(
        arm_joints=(
            "google:joint_torso",
            "google:joint_shoulder",
            "google:joint_bicep",
            "google:joint_elbow",
            "google:joint_forearm",
            "google:joint_wrist",
            "google:joint_gripper",
        ),
        arm_actuators=(
            "google:joint_torso_actuator",
            "google:joint_shoulder_actuator",
            "google:joint_bicep_actuator",
            "google:joint_elbow_actuator",
            "google:joint_forearm_actuator",
            "google:joint_wrist_actuator",
            "google:joint_gripper_actuator",
        ),
        finger_joints=(
            "google:joint_finger_right",
            "google:joint_finger_left",
        ),
        finger_actuators=(
            "google:joint_finger_right_actuator",
            "google:joint_finger_left_actuator",
        ),
        finger_contact_geoms=(
            GOOGLE_RIGHT_FINGER_GEOMS,
            GOOGLE_LEFT_FINGER_GEOMS,
        ),
        gripper_body="google:link_gripper",
        grip_site="google:gripper",
        top_down_rotation=GOOGLE_TOP_DOWN_ROTATION,
        # Menagerie's angular fingers close as the joint value increases.
        open_command=0.01,
        closed_command=1.30,
        close_step=0.003,
        carry_position=np.array((0.0, -0.78, 0.94)),
        # Deterministic IK branch for the top-down carry pose.  Starting from
        # the previous seed selected a folded-under solution whose forearm
        # crossed the base visual shell.  This seed selects the continuous,
        # collision-checked branch that reaches the same worktop poses.
        home_seed=np.array(
            (-0.088, -0.079, 0.929, 1.618, 0.064, 1.572, -3.872)
        ),
        # Menagerie's zero configuration folds the arm vertically inside the
        # base footprint and is the required empty-gripper navigation state.
        navigation_joints=np.zeros(7),
        # Start with regular cylindrical objects.  Irregular handles require
        # separate visual/contact calibration before they are exposed in UI.
        supported_objects=("sugar_jar",),
    ),
}


def mobile_profile(robot_name: str) -> MobileRobotProfile:
    try:
        return MOBILE_PROFILES[robot_name]
    except KeyError as error:
        raise ValueError(f"Robot '{robot_name}' has no mobile-motion profile") from error


def manipulation_profile(robot_name: str) -> ManipulationProfile:
    try:
        return MANIPULATION_PROFILES[robot_name]
    except KeyError as error:
        raise ValueError(
            f"Robot '{robot_name}' has no generic manipulation profile"
        ) from error
