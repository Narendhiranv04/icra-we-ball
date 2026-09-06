"""Deterministic tests for fixed-fixture geometry projection and MuJoCo body aliases."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_scenes.baselines.vilain_tamp.config import Domain
from mujoco_scenes.baselines.vilain_tamp.identity import (
    FIXED_FIXTURE_GEOMETRY_BODIES,
    fixed_entity_binding,
    resolve_geometry_entity_name,
)
from mujoco_scenes.baselines.vilain_tamp.live_observations import _create_scene
from mujoco_scenes.baselines.vilain_tamp.live_refinement import (
    LiveRefinementError,
    MuJoCoGeometryKernel,
    _resolve_articulation_joint,
)


class MockSceneWrapper:
    """Wrap a scene to provide the MuJoCoPlanningScene interface for refinement."""

    def __init__(self, scene: object) -> None:
        self.model = scene.model
        self.data = scene.data
        self.mujoco = mujoco


def test_resolve_geometry_entity_name_mappings() -> None:
    """Test that all required controller/symbolic fixture IDs resolve correctly."""
    # Workshop
    assert resolve_geometry_entity_name("LEFT_DRAWER") == "left_tool_drawer"
    assert resolve_geometry_entity_name("left_drawer") == "left_tool_drawer"
    assert resolve_geometry_entity_name("RIGHT_DRAWER") == "right_tool_drawer"
    assert resolve_geometry_entity_name("right_drawer") == "right_tool_drawer"
    assert resolve_geometry_entity_name("TOOL_CABINET") == "tool_cabinet"
    assert resolve_geometry_entity_name("tool_cabinet") == "tool_cabinet"
    assert resolve_geometry_entity_name("MAIN_WORKBENCH_ZONE") == "workbench"
    assert resolve_geometry_entity_name("main_workbench_zone_surface") == "workbench"
    assert resolve_geometry_entity_name("main_workbench") == "workbench"
    assert resolve_geometry_entity_name("workshop_frame_joint") == "workshop_frame_joint"

    # Kitchen
    assert resolve_geometry_entity_name("D1") == "drawer_D1_tray"
    assert resolve_geometry_entity_name("d1") == "drawer_D1_tray"
    assert resolve_geometry_entity_name("D2") == "drawer_D2_tray"
    assert resolve_geometry_entity_name("d2") == "drawer_D2_tray"
    assert resolve_geometry_entity_name("C1") == "cabinet_C1"
    assert resolve_geometry_entity_name("c1") == "cabinet_C1"
    assert resolve_geometry_entity_name("C2") == "cabinet_C2"
    assert resolve_geometry_entity_name("c2") == "cabinet_C2"
    assert resolve_geometry_entity_name("B1") == "box_B1"
    assert resolve_geometry_entity_name("b1") == "box_B1"
    assert resolve_geometry_entity_name("countertop") == "countertop"
    assert resolve_geometry_entity_name("serving_area") == "serving_area"

    # Living Room
    assert resolve_geometry_entity_name("staging") == "a2_staging"
    assert resolve_geometry_entity_name("a2_staging") == "a2_staging"
    assert resolve_geometry_entity_name("personal_table_left") == "a2_personal_left"
    assert resolve_geometry_entity_name("personal_table_right") == "a2_personal_right"
    assert resolve_geometry_entity_name("shared_table") == "a2_control_table"
    assert resolve_geometry_entity_name("a2_personal_left") == "a2_personal_left"
    assert resolve_geometry_entity_name("a2_personal_right") == "a2_personal_right"
    assert resolve_geometry_entity_name("a2_control_table") == "a2_control_table"

    # Idempotence on already-resolved names
    for target_geo in FIXED_FIXTURE_GEOMETRY_BODIES.values():
        assert resolve_geometry_entity_name(target_geo) == target_geo

    # Movable objects remain untouched
    assert (
        resolve_geometry_entity_name("workshop_medium_phillips_screw")
        == "workshop_medium_phillips_screw"
    )
    assert resolve_geometry_entity_name("ab3_deep_bowl") == "ab3_deep_bowl"
    assert resolve_geometry_entity_name("a2_drink_left") == "a2_drink_left"


def test_fixed_entity_binding_populates_both_identities() -> None:
    """Verify that fixed_entity_binding records both controller and geometry identities."""
    binding = fixed_entity_binding("left_drawer", "LEFT_DRAWER", broad_class="storage")
    assert binding.entity_name == "LEFT_DRAWER"
    assert binding.geometry_entity_name == "left_tool_drawer"

    binding_lr = fixed_entity_binding("staging", "staging", broad_class="location")
    assert binding_lr.entity_name == "staging"
    assert binding_lr.geometry_entity_name == "a2_staging"

    # Explicit override honored if given
    custom = fixed_entity_binding(
        "custom_bench", "MAIN_WORKBENCH", broad_class="surface",
        geometry_entity_name="custom_bench_body",
    )
    assert custom.entity_name == "MAIN_WORKBENCH"
    assert custom.geometry_entity_name == "custom_bench_body"


def test_workshop_scene_geometry_resolution() -> None:
    """Verify that all Workshop fixed fixtures resolve cleanly against real MuJoCo model."""
    scene = _create_scene(Domain.WORKSHOP, "F0_MANUAL_FIRST_ONE_REGION", robot="google", layout_seed=None)
    kernel = MuJoCoGeometryKernel()
    wrapper = MockSceneWrapper(scene)

    workshop_fixtures = [
        ("LEFT_DRAWER", "left_tool_drawer"),
        ("RIGHT_DRAWER", "right_tool_drawer"),
        ("TOOL_CABINET", "tool_cabinet"),
        ("MAIN_WORKBENCH_ZONE", "workbench"),
        ("workshop_frame_joint", "workshop_frame_joint"),
    ]

    for controller_id, expected_body in workshop_fixtures:
        expected_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, expected_body)
        assert expected_bid >= 0, f"Expected body {expected_body} must exist"

        # Resolve entity geometry via kernel
        geom = kernel._entity_geometry(wrapper, controller_id)
        assert geom.entity_name == controller_id
        assert geom.body_id == expected_bid
        assert np.all(np.isfinite(geom.centroid_m))
        assert np.all(np.isfinite(geom.aabb_min_m))
        assert np.all(np.isfinite(geom.aabb_max_m))
        assert all(high >= low for low, high in zip(geom.aabb_min_m, geom.aabb_max_m))

    # Test articulation joint lookup
    assert _resolve_articulation_joint(wrapper, "LEFT_DRAWER") >= 0
    assert _resolve_articulation_joint(wrapper, "RIGHT_DRAWER") >= 0
    assert _resolve_articulation_joint(wrapper, "TOOL_CABINET") >= 0

    # Test handle grasp site lookup
    assert len(kernel._named_grasp_positions(wrapper, "LEFT_DRAWER")) > 0
    assert len(kernel._named_grasp_positions(wrapper, "RIGHT_DRAWER")) > 0
    assert len(kernel._named_grasp_positions(wrapper, "TOOL_CABINET")) > 0


def test_kitchen_scene_geometry_resolution() -> None:
    """Verify that all Kitchen fixed fixtures resolve cleanly against real MuJoCo model."""
    scene = _create_scene(Domain.KITCHEN, "F0_ALL_VISIBLE", robot="google", layout_seed=None)
    kernel = MuJoCoGeometryKernel()
    wrapper = MockSceneWrapper(scene)

    kitchen_fixtures = [
        ("D1", "drawer_D1_tray"),
        ("D2", "drawer_D2_tray"),
        ("C1", "cabinet_C1"),
        ("C2", "cabinet_C2"),
        ("B1", "box_B1"),
        ("countertop", "countertop"),
        ("serving_area", "serving_area"),
    ]

    for controller_id, expected_body in kitchen_fixtures:
        expected_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, expected_body)
        assert expected_bid >= 0, f"Expected body {expected_body} must exist"

        geom = kernel._entity_geometry(wrapper, controller_id)
        assert geom.entity_name == controller_id
        assert geom.body_id == expected_bid
        assert np.all(np.isfinite(geom.centroid_m))
        assert np.all(np.isfinite(geom.aabb_min_m))
        assert np.all(np.isfinite(geom.aabb_max_m))

    # Test articulation joint lookup
    assert _resolve_articulation_joint(wrapper, "D1") >= 0
    assert _resolve_articulation_joint(wrapper, "D2") >= 0
    assert _resolve_articulation_joint(wrapper, "C1") >= 0
    assert _resolve_articulation_joint(wrapper, "C2") >= 0
    assert _resolve_articulation_joint(wrapper, "B1") >= 0

    # Test handle grasp site lookup
    assert len(kernel._named_grasp_positions(wrapper, "D1")) > 0
    assert len(kernel._named_grasp_positions(wrapper, "D2")) > 0
    assert len(kernel._named_grasp_positions(wrapper, "C1")) > 0
    assert len(kernel._named_grasp_positions(wrapper, "C2")) > 0
    assert len(kernel._named_grasp_positions(wrapper, "B1")) > 0


def test_living_room_scene_geometry_resolution() -> None:
    """Verify that all Living Room fixed fixtures resolve cleanly against real MuJoCo model."""
    scene = _create_scene(Domain.LIVING_ROOM, "F0_ALL_OBJECTS_IN_STAGING", robot="google", layout_seed=None)
    kernel = MuJoCoGeometryKernel()
    wrapper = MockSceneWrapper(scene)

    living_fixtures = [
        ("staging", "a2_staging"),
        ("a2_staging", "a2_staging"),
        ("personal_table_left", "a2_personal_left"),
        ("personal_table_right", "a2_personal_right"),
        ("shared_table", "a2_control_table"),
        ("a2_personal_left", "a2_personal_left"),
        ("a2_personal_right", "a2_personal_right"),
        ("a2_control_table", "a2_control_table"),
    ]

    for controller_id, expected_body in living_fixtures:
        expected_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, expected_body)
        assert expected_bid >= 0, f"Expected body {expected_body} must exist"

        geom = kernel._entity_geometry(wrapper, controller_id)
        assert geom.entity_name == controller_id
        assert geom.body_id == expected_bid
        assert np.all(np.isfinite(geom.centroid_m))
        assert np.all(np.isfinite(geom.aabb_min_m))
        assert np.all(np.isfinite(geom.aabb_max_m))


def test_missing_entity_raises_live_refinement_error_on_truly_missing() -> None:
    """Verify that genuinely nonexistent entity names still raise MISSING_SCENE_ENTITY."""
    scene = _create_scene(Domain.WORKSHOP, "F0_MANUAL_FIRST_ONE_REGION", robot="google", layout_seed=None)
    kernel = MuJoCoGeometryKernel()
    wrapper = MockSceneWrapper(scene)

    with pytest.raises(LiveRefinementError, match="MuJoCo body is missing") as exc_info:
        kernel._entity_geometry(wrapper, "completely_nonexistent_entity_name_123")
    assert exc_info.value.reason_code == "MISSING_SCENE_ENTITY"
