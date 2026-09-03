"""Phase-4 Workshop tool-cabinet hinge calibration.

The Workshop tool cabinet was originally authored with its hinge on the right
edge.  The Phase-4 robot approaches this cabinet from the right-side access
corridor, so the opened door occupied the same corridor needed for reaching
objects on the cabinet shelf.

This module mirrors the complete compiled door mechanism about the cabinet
centreline before any Phase-4 action is executed.  The result follows the same
kinematic convention as Kitchen C2: a left-edge hinge, a negative-Z hinge axis,
and a positive 0..1.57 rad actuator command.  Therefore positive OPEN motion
swings the free/right edge outward and from right to left, leaving the cabinet's
right-side manipulation aperture clear.

Only mechanism geometry is changed.  No object pose, storage membership,
planner assignment, Phase-3 action, or task state is modified.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


TOOL_CABINET_BODY = "tool_cabinet"
TOOL_CABINET_DOOR_BODY = "tool_cabinet_door"
TOOL_CABINET_HINGE = "tool_cabinet_door_hinge"
TOOL_CABINET_ACTUATOR = "tool_cabinet_door_actuator"
TOOL_CABINET_HANDLE_SITE = "tool_cabinet_door_handle_grasp"
TOOL_CABINET_OPEN_VALUE = 1.45


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(
            f"Workshop left-hinge calibration is missing required model object {name!r}"
        )
    return int(object_id)


def _door_child_geom_ids(model: mujoco.MjModel, door_body_id: int) -> tuple[int, ...]:
    first = int(model.body_geomadr[door_body_id])
    count = int(model.body_geomnum[door_body_id])
    return tuple(range(first, first + count))


def _door_child_site_ids(model: mujoco.MjModel, door_body_id: int) -> tuple[int, ...]:
    return tuple(
        site_id
        for site_id in range(model.nsite)
        if int(model.site_bodyid[site_id]) == door_body_id
    )


def _handle_world(scene: Any, handle_site_id: int) -> np.ndarray:
    return np.asarray(scene.data.site_xpos[handle_site_id], dtype=float).copy()


def configure_workshop_tool_cabinet_left_hinge(scene: Any) -> dict[str, Any]:
    """Mirror the compiled TOOL_CABINET door to a left-edge hinge.

    The Workshop scene builder already creates all welds and actuator IDs by
    name, so mirroring the door body, its local child geometry/sites, inertia
    centre, and hinge axis preserves every controller binding.  The actuator
    remains positive-range exactly as before.
    """

    cached = getattr(scene, "_phase4_tool_cabinet_left_hinge_audit", None)
    if isinstance(cached, dict):
        return cached

    model = scene.model
    data = scene.data
    cabinet_body_id = _required_id(
        model, mujoco.mjtObj.mjOBJ_BODY, TOOL_CABINET_BODY
    )
    door_body_id = _required_id(
        model, mujoco.mjtObj.mjOBJ_BODY, TOOL_CABINET_DOOR_BODY
    )
    hinge_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, TOOL_CABINET_HINGE)
    actuator_id = _required_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, TOOL_CABINET_ACTUATOR
    )
    handle_site_id = _required_id(
        model, mujoco.mjtObj.mjOBJ_SITE, TOOL_CABINET_HANDLE_SITE
    )

    hinge_qpos_adr = int(model.jnt_qposadr[hinge_id])
    hinge_dof_adr = int(model.jnt_dofadr[hinge_id])

    # This calibration must happen while the execution scene is still in its
    # closed initial state.  Applying it to an already-open door would change
    # mechanism geometry underneath an active manipulation trajectory.
    if abs(float(data.qpos[hinge_qpos_adr])) > 1e-6:
        raise RuntimeError(
            "Workshop tool-cabinet hinge calibration must run before OPEN"
        )

    original_door_local_position = model.body_pos[door_body_id].copy()
    original_hinge_axis = model.jnt_axis[hinge_id].copy()

    # Mirror the hinge body from +X (right edge) to -X (left edge).
    model.body_pos[door_body_id, 0] = -abs(float(model.body_pos[door_body_id, 0]))

    # Mirror all door-owned collision/visual geoms and sites in local X.  This
    # moves the panel to the right of the new hinge and moves the handle to the
    # free/right edge without changing names consumed by the controller.
    mirrored_geom_ids = _door_child_geom_ids(model, door_body_id)
    for geom_id in mirrored_geom_ids:
        model.geom_pos[geom_id, 0] *= -1.0
    mirrored_site_ids = _door_child_site_ids(model, door_body_id)
    for site_id in mirrored_site_ids:
        model.site_pos[site_id, 0] *= -1.0

    # MuJoCo has already compiled the door's inertial frame from its geoms.
    # Mirror the local COM as well so door dynamics remain physically symmetric
    # instead of leaving the mass centre on the old side of the hinge.
    model.body_ipos[door_body_id, 0] *= -1.0

    # Kitchen-C2 convention: positive actuator command around -Z.  This makes
    # the free right edge swing outward/right-to-left while retaining the
    # existing positive joint range and OPEN target value.
    model.jnt_axis[hinge_id] = np.array((0.0, 0.0, -1.0), dtype=float)

    data.qpos[hinge_qpos_adr] = 0.0
    data.qvel[hinge_dof_adr] = 0.0
    data.ctrl[actuator_id] = 0.0
    mujoco.mj_forward(model, data)
    closed_handle_world = _handle_world(scene, handle_site_id)

    # Audit the kinematic open pose without advancing simulation.  Restore the
    # exact closed state immediately afterward; Phase-4 still performs the real
    # OPEN using the robot-held handle trajectory.
    data.qpos[hinge_qpos_adr] = TOOL_CABINET_OPEN_VALUE
    mujoco.mj_forward(model, data)
    open_handle_world = _handle_world(scene, handle_site_id)
    data.qpos[hinge_qpos_adr] = 0.0
    data.qvel[hinge_dof_adr] = 0.0
    data.ctrl[actuator_id] = 0.0
    mujoco.mj_forward(model, data)

    cabinet_world = np.asarray(data.xpos[cabinet_body_id], dtype=float)
    hinge_world = np.asarray(data.xpos[door_body_id], dtype=float)

    # Fail closed if the authored mirror no longer clears the right-side
    # aperture: the closed handle begins to the right of the hinge, and after
    # opening it must move both leftward and outward toward the robot (-Y).
    if not (
        closed_handle_world[0] > hinge_world[0]
        and open_handle_world[0] < closed_handle_world[0]
        and open_handle_world[1] < closed_handle_world[1]
    ):
        raise RuntimeError(
            "Workshop left-hinge calibration produced the wrong door swing: "
            f"hinge={hinge_world.tolist()} closed_handle={closed_handle_world.tolist()} "
            f"open_handle={open_handle_world.tolist()}"
        )

    audit = {
        "mechanism": "TOOL_CABINET",
        "configuration": "LEFT_EDGE_HINGE_RIGHT_TO_LEFT_OUTWARD_SWING",
        "reference_convention": "KITCHEN_C2",
        "cabinet_world_position_m": cabinet_world.tolist(),
        "original_door_local_position_m": original_door_local_position.tolist(),
        "door_local_position_m": model.body_pos[door_body_id].tolist(),
        "original_hinge_axis": original_hinge_axis.tolist(),
        "hinge_axis": model.jnt_axis[hinge_id].tolist(),
        "closed_handle_world_m": closed_handle_world.tolist(),
        "predicted_open_handle_world_m": open_handle_world.tolist(),
        "opened_joint_value_rad": TOOL_CABINET_OPEN_VALUE,
        "mirrored_geom_count": len(mirrored_geom_ids),
        "mirrored_site_count": len(mirrored_site_ids),
        "planner_or_task_state_changed": False,
    }
    scene._phase4_tool_cabinet_left_hinge_audit = audit
    print(
        "[P4-WORKSHOP-CABINET] left-edge hinge configured; "
        f"axis={audit['hinge_axis']} "
        f"closed_handle=({closed_handle_world[0]:.3f}, {closed_handle_world[1]:.3f}) "
        f"open_handle=({open_handle_world[0]:.3f}, {open_handle_world[1]:.3f})",
        flush=True,
    )
    return audit
