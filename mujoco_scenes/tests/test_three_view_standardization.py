from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import yaml

from mujoco_scenes.geometry_checker import (
    CANONICAL_VIEWPOINT_ROLES,
    load_inspection_rig_config,
)
from mujoco_scenes.scene_loader import GOOGLE_BASE_POSE
from mujoco_scenes.workshop_phase1.inspection_controller import (
    WorkshopPhase1InspectionController,
)
from mujoco_scenes.workshop_scene import (
    WORKSHOP_GOOGLE_BASE_POSE,
    WORKSHOP_INSPECTION_RIG_CONFIG,
)


ROOT = Path(__file__).resolve().parents[1]


def _world_pose(region: dict, role: str) -> tuple[np.ndarray, np.ndarray]:
    camera = region["cameras"][role]
    return (
        np.asarray(region["rig_position_world_m"], dtype=float)
        + np.asarray(camera["position_offset_m"], dtype=float),
        np.asarray(region["target_world_m"], dtype=float)
        + np.asarray(camera["look_at_offset_m"], dtype=float),
    )


def test_kitchen_and_workshop_have_exact_canonical_roles():
    for path in (
        ROOT / "configs" / "inspection_rigs.yaml",
        WORKSHOP_INSPECTION_RIG_CONFIG,
    ):
        config = load_inspection_rig_config(path)
        assert tuple(config["camera_slots"]) == CANONICAL_VIEWPOINT_ROLES
        assert config["view_validation"]["minimum_valid_rig_cameras"] == 3
        assert config["view_validation"]["minimum_object_camera_count"] == 2
        assert all(
            set(region["cameras"]) == set(CANONICAL_VIEWPOINT_ROLES)
            for region in config["regions"].values()
        )


def test_living_room_is_one_initial_three_view_capture():
    config = yaml.safe_load(
        (ROOT / "configs" / "l2_integrated_region_function_rig.yaml").read_text()
    )
    assert set(config["camera_slots"]) == set(CANONICAL_VIEWPOINT_ROLES)
    assert set(config["capture"]["cameras"]) == set(CANONICAL_VIEWPOINT_ROLES)
    assert "inspection_sequence" not in config


def test_detail_orientation_follows_region_access_geometry():
    config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
    for region_id in ("LEFT_DRAWER", "RIGHT_DRAWER"):
        position, target = _world_pose(config["regions"][region_id], "DETAIL")
        ray = target - position
        assert -ray[2] / np.linalg.norm(ray) > 0.45
    position, target = _world_pose(config["regions"]["TOOL_CABINET"], "DETAIL")
    ray = target - position
    assert abs(ray[1]) > abs(ray[2])


def test_all_roles_are_distinct_and_left_right_straddle_target():
    config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
    for region in config["regions"].values():
        poses = [_world_pose(region, role)[0] for role in CANONICAL_VIEWPOINT_ROLES]
        assert len({tuple(np.round(pose, 8)) for pose in poses}) == 3
        target_x = float(region["target_world_m"][0])
        assert poses[0][0] < target_x < poses[1][0]


def test_single_view_alias_explicitly_selects_detail():
    source = inspect.getsource(
        WorkshopPhase1InspectionController._capture_and_process_stage
    )
    assert 'obs.camera_id == "DETAIL"' in source
    assert "raw_obs[0]" not in source


def test_workshop_pose_is_owned_and_shared_pose_is_unchanged():
    assert GOOGLE_BASE_POSE == {
        "pos": "0 -1.25 0.06205",
        "quat": "0.7071068 0 0 0.7071068",
    }
    assert WORKSHOP_GOOGLE_BASE_POSE["pos"] == "0 -0.75 0.06205"
    assert WORKSHOP_GOOGLE_BASE_POSE != GOOGLE_BASE_POSE
    base_xy = np.fromstring(WORKSHOP_GOOGLE_BASE_POSE["pos"], sep=" ")[:2]
    config = load_inspection_rig_config(WORKSHOP_INSPECTION_RIG_CONFIG)
    for region in config["regions"].values():
        minimum = np.asarray(region["inspection_volume"]["minimum_world_m"])
        maximum = np.asarray(region["inspection_volume"]["maximum_world_m"])
        assert not np.all((base_xy >= minimum[:2]) & (base_xy <= maximum[:2]))
