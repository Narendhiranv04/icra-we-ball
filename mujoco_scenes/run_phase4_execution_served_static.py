"""Kitchen Phase-4 runner with a terminal serving-state latch.

The normal physical controller is unchanged through:
  PLACE(utensil, bowl) -> PICK(bowl) -> PLACE(bowl, serving_area).
Only after the bowl itself has been physically placed successfully on the
serving table do we latch the already-achieved serving state.  Both the bowl
and its assigned utensil are fixed to the static serving-area body at their
current live poses and their collision geoms are suppressed.  This makes the
served pair a task-level terminal assembly: it cannot chatter, tip, or be
knocked off by later unrelated robot motions.  If either member is explicitly
PICKed later, both latches are released and the original collision masks are
restored before the physical PICK begins.

No qpos/qvel write is used to establish the served state; the latch is enabled
only after the original physical PLACE controller reports success.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

import mujoco
import numpy as np

from . import scene_loader
from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher
from .run_phase4_execution import main as _base_main


_ORIGINAL_BUILD_SCENE_XML = scene_loader.build_scene_xml
_ORIGINAL_PLACE = KitchenGroundTruthExecutionDispatcher.place
_ORIGINAL_PICK = KitchenGroundTruthExecutionDispatcher.pick
_PATCHED = False

# This is only a no-visible-snap guard after an already successful physical
# PLACE.  It is not a placement-success tolerance.
MAX_LATCH_POSITION_PROJECTION_M = 1.0e-3
MAX_LATCH_ORIENTATION_PROJECTION_RAD = 1.0e-2


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _served_weld_name(backend_body: str) -> str:
    return f"phase4_served_terminal__{_safe_name(backend_body)}"


def _inject_served_welds(xml: str) -> str:
    """Compile inactive serving-area welds for bowls and utensils."""
    root = ET.fromstring(xml)
    body_names = {
        body.get("name")
        for body in root.findall(".//body")
        if body.get("name")
    }
    if "serving_area" not in body_names:
        return xml

    tokens = ("bowl", "spoon", "fork", "utensil", "stirrer")
    payload_bodies = sorted(
        name
        for name in body_names
        if any(token in name.lower() for token in tokens)
        and not name.startswith("google:")
    )
    if not payload_bodies:
        return xml

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    existing = {item.get("name") for item in equality if item.get("name")}

    for backend in payload_bodies:
        name = _served_weld_name(backend)
        if name in existing:
            continue
        ET.SubElement(
            equality,
            "weld",
            {
                "name": name,
                "body1": "serving_area",
                "body2": backend,
                "active": "false",
                "solref": "0.01 1",
            },
        )
        existing.add(name)

    return ET.tostring(root, encoding="unicode")


def _patched_build_scene_xml(*args: Any, **kwargs: Any) -> str:
    return _inject_served_welds(_ORIGINAL_BUILD_SCENE_XML(*args, **kwargs))


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> tuple[int, ...]:
    first = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    return tuple(range(first, first + count))


def _store(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
) -> dict[str, dict[str, Any]]:
    value = getattr(dispatcher, "_phase4_served_terminal_latches", None)
    if value is None:
        value = {}
        dispatcher._phase4_served_terminal_latches = value
    return value


def _release_component(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
    planner_id: str,
) -> dict[str, Any] | None:
    state = _store(dispatcher).pop(planner_id, None)
    if state is None:
        return None

    model, data = dispatcher.scene.model, dispatcher.scene.data
    equality_id = int(state["equality_id"])
    if 0 <= equality_id < model.neq:
        data.eq_active[equality_id] = 0

    for geom_id, contype, conaffinity in state["geom_masks"]:
        model.geom_contype[int(geom_id)] = int(contype)
        model.geom_conaffinity[int(geom_id)] = int(conaffinity)

    mujoco.mj_forward(model, data)
    print(
        f"[P4-SERVED-LOCK] RELEASED {planner_id}({state['backend_body']})",
        flush=True,
    )
    return {
        "released": True,
        "planner_id": planner_id,
        "backend_body": state["backend_body"],
    }


def _release_pair_for_pick(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
) -> list[dict[str, Any]]:
    """If one member of a served pair is PICKed, restore both members."""
    store = _store(dispatcher)
    pair_ids: set[str] = set()
    for state in store.values():
        pair = tuple(state.get("pair_ids", ()))
        if object_id in pair:
            pair_ids.update(str(item) for item in pair)
    if object_id in store:
        pair_ids.add(object_id)

    released: list[dict[str, Any]] = []
    for planner_id in sorted(pair_ids):
        item = _release_component(dispatcher, planner_id)
        if item is not None:
            released.append(item)
    return released


def _activate_static_component(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
    planner_id: str,
    *,
    pair_ids: tuple[str, str],
    role: str,
) -> dict[str, Any]:
    """Fix one already-served payload to serving_area at its current pose."""
    model, data = dispatcher.scene.model, dispatcher.scene.data
    binding = dispatcher.binding_by_id.get(planner_id)
    backend = binding.get("physical_backend_body") if binding else None
    if not backend:
        raise RuntimeError(f"SERVED_LOCK: missing backend for {planner_id}")

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend)
    serving_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "serving_area"
    )
    if body_id < 0 or serving_body < 0:
        raise RuntimeError(
            f"SERVED_LOCK: missing body for {planner_id}->{backend} or serving_area"
        )

    equality_name = _served_weld_name(backend)
    equality_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
    )
    if equality_id < 0:
        raise RuntimeError(
            "SERVED_LOCK: equality not compiled: " + equality_name
        )

    _release_component(dispatcher, planner_id)

    before_pos = data.xpos[body_id].copy()
    before_quat = data.xquat[body_id].copy()

    inv_pos = np.empty(3)
    inv_quat = np.empty(4)
    rel_pos = np.empty(3)
    rel_quat = np.empty(4)
    mujoco.mju_negPose(
        inv_pos,
        inv_quat,
        data.xpos[serving_body],
        data.xquat[serving_body],
    )
    mujoco.mju_mulPose(
        rel_pos,
        rel_quat,
        inv_pos,
        inv_quat,
        data.xpos[body_id],
        data.xquat[body_id],
    )
    model.eq_data[equality_id, 3:6] = rel_pos
    model.eq_data[equality_id, 6:10] = rel_quat

    # Terminal served objects no longer participate in unrelated future
    # manipulation contacts.  Preserve the exact masks so a later explicit
    # PICK can restore normal dynamics before grasping.
    masks: list[tuple[int, int, int]] = []
    for geom_id in _body_geom_ids(model, body_id):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        masks.append((geom_id, contype, conaffinity))
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    data.eq_active[equality_id] = 1
    mujoco.mj_forward(model, data)

    pos_delta = float(np.linalg.norm(data.xpos[body_id] - before_pos))
    quat_dot = abs(float(np.dot(data.xquat[body_id], before_quat)))
    angle_delta = 2.0 * float(
        np.arccos(np.clip(quat_dot, -1.0, 1.0))
    )

    if (
        pos_delta > MAX_LATCH_POSITION_PROJECTION_M
        or angle_delta > MAX_LATCH_ORIENTATION_PROJECTION_RAD
    ):
        data.eq_active[equality_id] = 0
        for geom_id, contype, conaffinity in masks:
            model.geom_contype[geom_id] = contype
            model.geom_conaffinity[geom_id] = conaffinity
        mujoco.mj_forward(model, data)
        raise RuntimeError(
            "SERVED_LOCK: activation projection too large for "
            f"{planner_id}: {pos_delta:.6g} m / {angle_delta:.6g} rad"
        )

    _store(dispatcher)[planner_id] = {
        "backend_body": backend,
        "equality_id": equality_id,
        "equality_name": equality_name,
        "geom_masks": masks,
        "pair_ids": pair_ids,
        "role": role,
    }

    print(
        f"[P4-SERVED-LOCK] ACTIVE {role} {planner_id}({backend}); "
        f"delta={pos_delta:.3e} m / {angle_delta:.3e} rad",
        flush=True,
    )
    return {
        "active": True,
        "mode": "TERMINAL_SERVED_STATE_LATCH",
        "role": role,
        "planner_id": planner_id,
        "backend_body": backend,
        "equality_name": equality_name,
        "collision_suppressed_while_served": True,
        "direct_payload_pose_write": False,
        "direct_payload_velocity_write": False,
        "activation_position_delta_m": pos_delta,
        "activation_orientation_delta_rad": angle_delta,
    }


def _patched_place(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
) -> dict[str, Any]:
    # The complete physical action remains the known-good original controller.
    result = _ORIGINAL_PLACE(self, object_id, destination)
    if not result.get("success", False):
        return result

    bowl_id = None
    tool_id = None
    if destination == "serving_area":
        tool_id = self.assignment.soup_utensils_by_target.get(object_id)
        if tool_id:
            bowl_id = object_id
    elif destination in self.assignment.soup_utensils_by_target:
        bowl_dest = self.inventory_by_id.get(destination, {}).get("location")
        if bowl_dest == "serving_area" and self.assignment.soup_utensils_by_target.get(destination) == object_id:
            bowl_id = destination
            tool_id = object_id

    if not bowl_id or not tool_id:
        return result

    tool_loc = self.inventory_by_id.get(tool_id, {}).get("location")
    tool_is_served = bool(tool_loc in (bowl_id, "serving_area"))
    bowl_loc = self.inventory_by_id.get(bowl_id, {}).get("location")
    bowl_is_served = bool(bowl_loc == "serving_area")

    pair_ids = (bowl_id, tool_id)
    activated: list[str] = []
    try:
        bowl_lock = None
        if bowl_is_served:
            bowl_lock = _activate_static_component(
                self,
                bowl_id,
                pair_ids=pair_ids,
                role="SOUP_BOWL",
            )
            activated.append(bowl_id)
        utensil_lock = None
        if tool_is_served:
            utensil_lock = _activate_static_component(
                self,
                tool_id,
                pair_ids=pair_ids,
                role="SOUP_UTENSIL",
            )
            activated.append(tool_id)
    except Exception as error:
        # Never leave a half-latched pair if setup itself fails.
        for planner_id in reversed(activated):
            _release_component(self, planner_id)
        message = f"{type(error).__name__}: {error}"
        print(f"[P4-SERVED-LOCK] FAILED: {message}", flush=True)
        return {
            **result,
            "success": False,
            "status": "SERVED_TERMINAL_LATCH_FAILED",
            "failure_code": "EXECUTION_ERROR",
            "message": message,
        }

    return {
        **result,
        "served_terminal_state_latched": bool(activated),
        "served_bowl_lock": bowl_lock,
        "served_utensil_lock": utensil_lock,
    }


def _patched_pick(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
) -> dict[str, Any]:
    released = _release_pair_for_pick(self, object_id)
    result = _ORIGINAL_PICK(self, object_id)
    if released:
        result = {**result, "released_served_terminal_latches": released}
    return result


def install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    scene_loader.build_scene_xml = _patched_build_scene_xml
    KitchenGroundTruthExecutionDispatcher.place = _patched_place
    KitchenGroundTruthExecutionDispatcher.pick = _patched_pick
    _PATCHED = True


def main() -> int:
    install_patch()
    return _base_main()


if __name__ == "__main__":
    raise SystemExit(main())
