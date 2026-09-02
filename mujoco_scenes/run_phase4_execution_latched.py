"""Phase-4 Kitchen runner with a post-placement containment latch.

This keeps the known-good Phase-4 controllers untouched.  A soup utensil must
first be physically placed and accepted by the existing containment checks.
Only after that successful PLACE is the live utensil pose rigidly attached to
its assigned bowl.  The bowl remains a normal free MuJoCo body, so later PICK
and PLACE actions move the bowl-plus-utensil assembly naturally.
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


def _nested_weld_name(tool_backend: str, bowl_backend: str) -> str:
    return (
        "phase4_nested_containment__"
        f"{_safe_name(tool_backend)}__in__{_safe_name(bowl_backend)}"
    )


def _inject_nested_containment_welds(xml: str) -> str:
    """Compile inactive utensil<->bowl welds without changing normal contacts."""
    root = ET.fromstring(xml)
    body_names = {
        body.get("name")
        for body in root.findall(".//body")
        if body.get("name")
    }
    utensil_tokens = ("spoon", "fork", "utensil", "stirrer")
    tool_bodies = sorted(
        name for name in body_names
        if any(token in name.lower() for token in utensil_tokens)
        and not name.startswith("google:")
    )
    bowl_bodies = sorted(
        name for name in body_names
        if "bowl" in name.lower() and not name.startswith("google:")
    )
    if not tool_bodies or not bowl_bodies:
        return xml

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    existing = {
        item.get("name") for item in equality
        if item.get("name")
    }
    for tool in tool_bodies:
        for bowl in bowl_bodies:
            if tool == bowl:
                continue
            name = _nested_weld_name(tool, bowl)
            if name in existing:
                continue
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": name,
                    "body1": bowl,
                    "body2": tool,
                    "active": "false",
                },
            )
            existing.add(name)
    return ET.tostring(root, encoding="unicode")


def _patched_build_scene_xml(*args: Any, **kwargs: Any) -> str:
    return _inject_nested_containment_welds(
        _ORIGINAL_BUILD_SCENE_XML(*args, **kwargs)
    )


def _soup_pair(dispatcher: KitchenGroundTruthExecutionDispatcher,
               tool_id: str, bowl_id: str) -> bool:
    return any(
        row.get("tool_instance") == tool_id
        and row.get("target_instance") == bowl_id
        for row in dispatcher.assignment.soup_assignments
    )


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> tuple[int, ...]:
    first = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    return tuple(range(first, first + count))


def _latch_store(dispatcher: KitchenGroundTruthExecutionDispatcher) -> dict[str, dict[str, Any]]:
    store = getattr(dispatcher, "_phase4_nested_containment_latches", None)
    if store is None:
        store = {}
        dispatcher._phase4_nested_containment_latches = store
    return store


def _release_nested_latch(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
    tool_id: str,
) -> dict[str, Any] | None:
    """Restore a latched utensil to an independently dynamic body before PICK."""
    store = _latch_store(dispatcher)
    state = store.pop(tool_id, None)
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
    return {
        "released": True,
        "tool_id": tool_id,
        "bowl_id": state["bowl_id"],
        "equality_name": state["equality_name"],
    }


def _activate_nested_latch(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
    tool_id: str,
    bowl_id: str,
) -> dict[str, Any]:
    """Latch the already-verified live utensil pose to its assigned bowl."""
    model, data = dispatcher.scene.model, dispatcher.scene.data
    tool_backend = dispatcher.binding_by_id[tool_id]["physical_backend_body"]
    bowl_backend = dispatcher.binding_by_id[bowl_id]["physical_backend_body"]
    tool_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, tool_backend
    )
    bowl_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, bowl_backend
    )
    if tool_body < 0 or bowl_body < 0:
        raise RuntimeError(
            "POST_PLACEMENT_CONTAINMENT_LATCH: resolved body is missing"
        )
    equality_name = _nested_weld_name(tool_backend, bowl_backend)
    equality_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
    )
    if equality_id < 0:
        raise RuntimeError(
            "POST_PLACEMENT_CONTAINMENT_LATCH: equality was not compiled: "
            + equality_name
        )

    # If this exact tool was previously latched, release the old relation first.
    _release_nested_latch(dispatcher, tool_id)

    before_tool_pos = data.xpos[tool_body].copy()
    before_tool_quat = data.xquat[tool_body].copy()
    inverse_pos = np.empty(3)
    inverse_quat = np.empty(4)
    relative_pos = np.empty(3)
    relative_quat = np.empty(4)
    mujoco.mju_negPose(
        inverse_pos,
        inverse_quat,
        data.xpos[bowl_body],
        data.xquat[bowl_body],
    )
    mujoco.mju_mulPose(
        relative_pos,
        relative_quat,
        inverse_pos,
        inverse_quat,
        data.xpos[tool_body],
        data.xquat[tool_body],
    )

    # MuJoCo weld data stores the body2 pose relative to body1 in 3:10.
    # This mirrors the existing grasp-weld implementation and does not write
    # either payload's qpos or qvel.
    model.eq_data[equality_id, 3:6] = relative_pos
    model.eq_data[equality_id, 6:10] = relative_quat

    # Once containment has been physically established, the utensil is a rigid
    # part of the served bowl assembly.  Suppress the utensil's own contact
    # geoms while latched so its already-satisfied bowl contacts cannot chatter
    # against the weld.  Original masks are restored if the utensil is PICKed.
    tool_geom_masks: list[tuple[int, int, int]] = []
    for geom_id in _body_geom_ids(model, tool_body):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        tool_geom_masks.append((geom_id, contype, conaffinity))
        if contype or conaffinity:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0

    data.eq_active[equality_id] = 1
    mujoco.mj_forward(model, data)

    # mj_forward must not snap the body because the weld was configured from
    # the current live relative pose.  This is a diagnostic invariant only.
    position_delta = float(np.linalg.norm(data.xpos[tool_body] - before_tool_pos))
    quat_dot = abs(float(np.dot(data.xquat[tool_body], before_tool_quat)))
    orientation_delta = 2.0 * float(np.arccos(np.clip(quat_dot, -1.0, 1.0)))
    if position_delta > 1e-9 or orientation_delta > 1e-8:
        data.eq_active[equality_id] = 0
        for geom_id, contype, conaffinity in tool_geom_masks:
            model.geom_contype[geom_id] = contype
            model.geom_conaffinity[geom_id] = conaffinity
        mujoco.mj_forward(model, data)
        raise RuntimeError(
            "POST_PLACEMENT_CONTAINMENT_LATCH: activation changed live pose"
        )

    state = {
        "bowl_id": bowl_id,
        "tool_backend": tool_backend,
        "bowl_backend": bowl_backend,
        "equality_id": equality_id,
        "equality_name": equality_name,
        "tool_geom_masks": tool_geom_masks,
        "relative_position_m": tuple(map(float, relative_pos)),
        "relative_orientation_wxyz": tuple(map(float, relative_quat)),
    }
    _latch_store(dispatcher)[tool_id] = state
    return {
        "active": True,
        "mode": "POST_PLACEMENT_CONTAINMENT_LATCH",
        "tool_id": tool_id,
        "bowl_id": bowl_id,
        "tool_backend": tool_backend,
        "bowl_backend": bowl_backend,
        "equality_name": equality_name,
        "relative_position_m": list(map(float, relative_pos)),
        "relative_orientation_wxyz": list(map(float, relative_quat)),
        "tool_collision_suppressed_while_latched": True,
        "direct_payload_pose_write": False,
        "direct_payload_velocity_write": False,
        "activation_position_delta_m": position_delta,
        "activation_orientation_delta_rad": orientation_delta,
    }


def _patched_place(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
) -> dict[str, Any]:
    result = _ORIGINAL_PLACE(self, object_id, destination)
    if not result.get("success", False):
        return result
    if destination not in self.binding_by_id:
        return result
    if not _soup_pair(self, object_id, destination):
        return result
    if result.get("direct_payload_pose_write", False):
        return {
            **result,
            "success": False,
            "status": "POST_PLACEMENT_CONTAINMENT_LATCH_REJECTED",
            "failure_code": "EXECUTION_ERROR",
            "message": "Containment latch refuses assisted/direct-pose placement",
        }
    try:
        latch = _activate_nested_latch(self, object_id, destination)
    except Exception as error:
        return {
            **result,
            "success": False,
            "status": "POST_PLACEMENT_CONTAINMENT_LATCH_FAILED",
            "failure_code": "EXECUTION_ERROR",
            "message": str(error),
        }
    return {
        **result,
        "nested_containment_latch": latch,
        "post_placement_containment_latched": True,
    }


def _patched_pick(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
) -> dict[str, Any]:
    released = _release_nested_latch(self, object_id)
    result = _ORIGINAL_PICK(self, object_id)
    if released is not None:
        result = {**result, "released_nested_containment_latch": released}
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
