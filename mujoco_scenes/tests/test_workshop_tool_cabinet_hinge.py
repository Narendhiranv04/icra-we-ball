import mujoco
import numpy as np
import pytest

from mujoco_scenes.workshop_scene import WorkshopScene
from mujoco_scenes.workshop_tool_cabinet_hinge import (
    TOOL_CABINET_HANDLE_SITE,
    TOOL_CABINET_HINGE,
    TOOL_CABINET_OPEN_VALUE,
    configure_workshop_tool_cabinet_left_hinge,
)


CABINET_REACH_VARIANTS = (
    "F5_POWER_FIRST_THREE_REGIONS",
    "F6_MANUAL_ONLY",
    "F7_POWER_ONLY",
)


@pytest.mark.parametrize("variant", CABINET_REACH_VARIANTS)
def test_w6_w8_tool_cabinet_is_left_hinged_and_swings_out_of_right_access_corridor(variant):
    scene = WorkshopScene(robot="none", variant=variant)
    model = scene.model
    data = scene.data

    door_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "tool_cabinet_door"
    )
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TOOL_CABINET_HINGE)
    handle = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, TOOL_CABINET_HANDLE_SITE
    )
    assert door_body >= 0 and hinge >= 0 and handle >= 0

    # The base template historically used the opposite/right-edge hinge.
    assert model.body_pos[door_body, 0] > 0.0
    assert np.allclose(model.jnt_axis[hinge], (0.0, 0.0, 1.0))

    audit = configure_workshop_tool_cabinet_left_hinge(scene)
    assert audit["configuration"] == "LEFT_EDGE_HINGE_RIGHT_TO_LEFT_OUTWARD_SWING"
    assert audit["reference_convention"] == "KITCHEN_C2"
    assert not audit["planner_or_task_state_changed"]

    # Mirrored mechanism: hinge on cabinet left edge, free handle on the right,
    # positive OPEN command around -Z exactly like Kitchen C2.
    assert model.body_pos[door_body, 0] < 0.0
    assert np.allclose(model.jnt_axis[hinge], (0.0, 0.0, -1.0))
    assert model.site_pos[handle, 0] > 0.0

    qpos_adr = int(model.jnt_qposadr[hinge])
    closed_handle = data.site_xpos[handle].copy()
    hinge_world = data.xpos[door_body].copy()
    assert closed_handle[0] > hinge_world[0]

    data.qpos[qpos_adr] = TOOL_CABINET_OPEN_VALUE
    mujoco.mj_forward(model, data)
    opened_handle = data.site_xpos[handle].copy()

    # The free/right edge must swing leftward and outward (-Y), so the robot's
    # right-side cabinet approach is no longer occupied by the open door.
    assert opened_handle[0] < closed_handle[0]
    assert opened_handle[1] < closed_handle[1]
    assert opened_handle[0] < 0.40  # opened door stays on the cabinet's left side


@pytest.mark.parametrize("variant", CABINET_REACH_VARIANTS)
def test_left_hinge_calibration_is_idempotent_and_preserves_closed_start(variant):
    scene = WorkshopScene(robot="none", variant=variant)
    first = configure_workshop_tool_cabinet_left_hinge(scene)
    second = configure_workshop_tool_cabinet_left_hinge(scene)

    assert second == first
    hinge = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_JOINT, TOOL_CABINET_HINGE
    )
    assert abs(float(scene.data.qpos[scene.model.jnt_qposadr[hinge]])) < 1e-9
    assert not scene.state.container_open_state["TOOL_CABINET"]
