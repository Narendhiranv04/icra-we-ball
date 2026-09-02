"""Kitchen Phase-4 runner that freezes soup utensils only after serving.

The normal physical controller is unchanged through:
  PLACE(utensil, bowl) -> PICK(bowl) -> PLACE(bowl, serving_area).
After the bowl has been physically placed on the serving table, the assigned
utensil is fixed to the static serving-area body at its current live world pose
and its contacts are disabled. This removes the delayed spoon chatter without
adding spoon weight/torque to the bowl. If that bowl or utensil is later PICKed,
the serving latch is released first and the utensil collision masks are restored.
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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _served_weld_name(tool_backend: str) -> str:
    return f"phase4_served_static__{_safe_name(tool_backend)}"


def _inject_served_welds(xml: str) -> str:
    root = ET.fromstring(xml)
    body_names = {
        body.get("name")
        for body in root.findall(".//body")
        if body.get("name")
    }
    if "serving_area" not in body_names:
        return xml
    utensil_tokens = ("spoon", "fork", "utensil", "stirrer")
    tool_bodies = sorted(
        name for name in body_names
        if any(token in name.lower() for token in utensil_tokens)
        and not name.startswith("google:")
    )
    if not tool_bodies:
        return xml
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    existing = {item.get("name") for item in equality if item.get("name")}
    for tool in tool_bodies:
        name = _served_weld_name(tool)
        if name in existing:
            continue
        ET.SubElement(
            equality,
            "weld",
            {
                "name": name,
                "body1": "serving_area",
                "body2": tool,
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


def _store(dispatcher: KitchenGroundTruthExecutionDispatcher) -> dict[str, dict[str, Any]]:
    value = getattr(dispatcher, "_phase4_served_static_latches", None)
    if value is None:
        value = {}
        dispatcher._phase4_served_static_latches = value
    return value


def _release_tool(dispatcher: KitchenGroundTruthExecutionDispatcher, tool_id: str) -> dict[str, Any] | None:
    state = _store(dispatcher).pop(tool_id, None)
    if state is None:
        return None
    model, data = dispatcher.scene.model, dispatcher.scene.data
    equality_id = int(state["equality_id"])
    if 0 <= equality_id < model.neq:
        data.eq_active[equality_id] = 0
    for geom_id, contype, conaffinity in state["tool_geom_masks"]:
        model.geom_contype[int(geom_id)] = int(contype)
        model.geom_conaffinity[int(geom_id)] = int(conaffinity)
    mujoco.mj_forward(model, data)
    print(
        f"[P4-SERVED-STATIC] RELEASED {tool_id} from {state['bowl_id']}",
        flush=True,
    )
    return {"released": True, "tool_id": tool_id, "bowl_id": state["bowl_id"]}


def _release_for_pick(dispatcher: KitchenGroundTruthExecutionDispatcher, object_id: str) -> list[dict[str, Any]]:
    released: list[dict[str, Any]] = []
    # Explicitly picking the utensil releases its latch.
    state = _store(dispatcher).get(object_id)
    if state is not None:
        item = _release_tool(dispatcher, object_id)
        if item:
            released.append(item)
    # Explicitly picking a served bowl also releases its assigned utensil first.
    for tool_id, tool_state in list(_store(dispatcher).items()):
        if tool_state.get("bowl_id") == object_id:
            item = _release_tool(dispatcher, tool_id)
            if item:
                released.append(item)
    return released


def _activate_served_static(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
    bowl_id: str,
    tool_id: str,
) -> dict[str, Any]:
    model, data = dispatcher.scene.model, dispatcher.scene.data
    tool_backend = dispatcher.binding_by_id[tool_id]["physical_backend_body"]
    tool_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool_backend)
    serving_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "serving_area")
    if tool_body < 0 or serving_body < 0:
        raise RuntimeError("SERVED_STATIC_LATCH: missing tool or serving_area body")

    equality_name = _served_weld_name(tool_backend)
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
    if equality_id < 0:
        raise RuntimeError("SERVED_STATIC_LATCH: equality not compiled: " + equality_name)

    _release_tool(dispatcher, tool_id)

    before_pos = data.xpos[tool_body].copy()
    before_quat = data.xquat[tool_body].copy()

    inv_pos = np.empty(3)
    inv_quat = np.empty(4)
    rel_pos = np.empty(3)
    rel_quat = np.empty(4)
    mujoco.mju_negPose(inv_pos, inv_quat, data.xpos[serving_body], data.xquat[serving_body])
    mujoco.mju_mulPose(
        rel_pos,
        rel_quat,
        inv_pos,
        inv_quat,
        data.xpos[tool_body],
        data.xquat[tool_body],
    )
    model.eq_data[equality_id, 3:6] = rel_pos
    model.eq_data[equality_id, 6:10] = rel_quat

    masks: list[tuple[int, int, int]] = []
    for geom_id in _body_geom_ids(model, tool_body):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        masks.append((geom_id, contype, conaffinity))
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    data.eq_active[equality_id] = 1
    mujoco.mj_forward(model, data)

    pos_delta = float(np.linalg.norm(data.xpos[tool_body] - before_pos))
    quat_dot = abs(float(np.dot(data.xquat[tool_body], before_quat)))
    angle_delta = 2.0 * float(np.arccos(np.clip(quat_dot, -1.0, 1.0)))
    if pos_delta > 5e-4 or angle_delta > 5e-3:
        data.eq_active[equality_id] = 0
        for geom_id, contype, conaffinity in masks:
            model.geom_contype[geom_id] = contype
            model.geom_conaffinity[geom_id] = conaffinity
        mujoco.mj_forward(model, data)
        raise RuntimeError(
            "SERVED_STATIC_LATCH: activation snap too large: "
            f"{pos_delta:.6g} m / {angle_delta:.6g} rad"
        )

    _store(dispatcher)[tool_id] = {
        "bowl_id": bowl_id,
        "tool_backend": tool_backend,
        "equality_id": equality_id,
        "equality_name": equality_name,
        "tool_geom_masks": masks,
    }
    print(
        f"[P4-SERVED-STATIC] ACTIVE {tool_id}({tool_backend}) after serving {bowl_id}; "
        f"delta={pos_delta:.3e} m / {angle_delta:.3e} rad",
        flush=True,
    )
    return {
        "active": True,
        "mode": "SERVED_UTENSIL_STATIC_LATCH",
        "tool_id": tool_id,
        "bowl_id": bowl_id,
        "equality_name": equality_name,
        "tool_collision_suppressed": True,
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
    # Everything through the bowl's actual serving placement is the original
    # known-good physical controller. Nothing is latched at PLACE(tool, bowl).
    result = _ORIGINAL_PLACE(self, object_id, destination)
    if not result.get("success", False) or destination != "serving_area":
        return result

    tool_id = self.assignment.soup_utensils_by_target.get(object_id)
    if not tool_id:
        return result
    try:
        latch = _activate_served_static(self, object_id, tool_id)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        print(f"[P4-SERVED-STATIC] FAILED: {message}", flush=True)
        return {
            **result,
            "success": False,
            "status": "SERVED_STATIC_LATCH_FAILED",
            "failure_code": "EXECUTION_ERROR",
            "message": message,
        }
    return {
        **result,
        "served_utensil_static_latch": latch,
        "served_utensil_static_latched": True,
    }


def _patched_pick(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
) -> dict[str, Any]:
    released = _release_for_pick(self, object_id)
    result = _ORIGINAL_PICK(self, object_id)
    if released:
        result = {**result, "released_served_static_latches": released}
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
