"""Kitchen Phase-4 authored-state preservation across long inspection sequences.

Feasible Kitchen variants can execute several OPEN/navigation actions before the
first task manipulation. Countertop payloads are free bodies, so those extra
physics seconds can let authored spoon/bowl poses drift away from the calibrated
layout before the task begins.

Initially observed countertop payloads are therefore held at their authored
current poses while they are untouched. A hold is released at the first action
that physically uses that payload:
  * before PICK(payload), or
  * before PLACE(other_payload, payload) when it is the relative destination.

This keeps untouched scene objects stable through inspection and unrelated
preceding actions, but guarantees that a bowl is already a normal free body
before a utensil is physically placed into it. No object qpos/qvel is written
and no collision geometry is disabled.

The generic grasp planner also distinguishes the non-contact ``Approaching
above object`` waypoint from the final contact pose. Routed approaches are
labelled ``Approaching above object route N``; the generic tolerance selector
matches the canonical label exactly, so routed waypoints can accidentally fall
back to the strict terminal IK tolerance. Kitchen tabletop bowls use a bounded
2 cm / 8 degree intermediate envelope and routed approach labels are
canonicalized only while solving that non-contact waypoint. Final descent,
bilateral contact, attachment, and held-state checks remain unchanged.
"""

from __future__ import annotations

from dataclasses import replace
import re
import xml.etree.ElementTree as ET
from typing import Any

import mujoco
import numpy as np

from . import scene_loader
from .generic_manipulation import CalibratedPickPlaceExecutor
from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher


_ORIGINAL_BUILD_SCENE_XML = scene_loader.build_scene_xml
_ORIGINAL_INIT = KitchenGroundTruthExecutionDispatcher.__init__
_ORIGINAL_PICK = KitchenGroundTruthExecutionDispatcher.pick
_ORIGINAL_PLACE = KitchenGroundTruthExecutionDispatcher.place
_ORIGINAL_SOLVE_POINTS = CalibratedPickPlaceExecutor._solve_points
_PATCHED = False

_TABLE_BOWL_INTERMEDIATE_POSITION_TOLERANCE_M = 0.020
_TABLE_BOWL_INTERMEDIATE_ORIENTATION_TOLERANCE_RAD = float(np.deg2rad(8.0))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _counter_weld_name(backend_body: str) -> str:
    return f"phase4_countertop_presentation__{_safe_name(backend_body)}"


def _is_free_payload_body(body: ET.Element) -> bool:
    if body.find("freejoint") is not None:
        return True
    return any(
        child.tag == "joint" and child.get("type") == "free"
        for child in list(body)
    )


def _inject_countertop_presentation_welds(xml: str) -> str:
    """Compile inactive countertop welds for every free payload body."""
    root = ET.fromstring(xml)
    body_names = {
        body.get("name")
        for body in root.findall(".//body")
        if body.get("name")
    }
    if "countertop" not in body_names:
        return xml

    payload_bodies = sorted(
        body.get("name")
        for body in root.findall(".//body")
        if body.get("name")
        and not str(body.get("name")).startswith("google:")
        and _is_free_payload_body(body)
    )
    if not payload_bodies:
        return xml

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    existing = {item.get("name") for item in equality if item.get("name")}

    for backend in payload_bodies:
        assert backend is not None
        name = _counter_weld_name(backend)
        if name in existing:
            continue
        ET.SubElement(
            equality,
            "weld",
            {
                "name": name,
                "body1": "countertop",
                "body2": backend,
                "active": "false",
                "solref": "0.01 1",
            },
        )
        existing.add(name)

    return ET.tostring(root, encoding="unicode")


def _patched_build_scene_xml(*args: Any, **kwargs: Any) -> str:
    return _inject_countertop_presentation_welds(
        _ORIGINAL_BUILD_SCENE_XML(*args, **kwargs)
    )


def _presentation_store(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
) -> dict[str, dict[str, Any]]:
    value = getattr(dispatcher, "_phase4_countertop_presentation_latches", None)
    if value is None:
        value = {}
        dispatcher._phase4_countertop_presentation_latches = value
    return value


def _set_weld_to_current_relative_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    equality_id: int,
    parent_body_id: int,
    child_body_id: int,
) -> None:
    inv_pos = np.empty(3)
    inv_quat = np.empty(4)
    rel_pos = np.empty(3)
    rel_quat = np.empty(4)
    mujoco.mju_negPose(
        inv_pos,
        inv_quat,
        data.xpos[parent_body_id],
        data.xquat[parent_body_id],
    )
    mujoco.mju_mulPose(
        rel_pos,
        rel_quat,
        inv_pos,
        inv_quat,
        data.xpos[child_body_id],
        data.xquat[child_body_id],
    )
    model.eq_data[equality_id, 3:6] = rel_pos
    model.eq_data[equality_id, 6:10] = rel_quat


def _configure_table_bowl_intermediate_ik(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
) -> None:
    """Keep routed bowl approach tolerance separate from final grasp tolerance."""
    executor = dispatcher.phase_b.manipulation.executor
    adjusted: list[tuple[str, str]] = []
    for planner_id, row in dispatcher.inventory_by_id.items():
        context = row.get("source_context") or {}
        if str(context.get("source_kind")) != "TABLE":
            continue
        binding = dispatcher.binding_by_id.get(planner_id) or {}
        if str(binding.get("grasp_family")) != "BOWL":
            continue
        backend = binding.get("physical_backend_body")
        if not backend:
            continue
        spec = executor.pick_specs.get(str(backend))
        if spec is None:
            continue

        current_position = spec.intermediate_ik_position_tolerance
        current_angle = spec.intermediate_ik_angle_tolerance_rad
        bounded_position = max(
            float(current_position or 0.0),
            _TABLE_BOWL_INTERMEDIATE_POSITION_TOLERANCE_M,
        )
        bounded_angle = max(
            float(current_angle or 0.0),
            _TABLE_BOWL_INTERMEDIATE_ORIENTATION_TOLERANCE_RAD,
        )
        executor.pick_specs[str(backend)] = replace(
            spec,
            intermediate_ik_position_tolerance=bounded_position,
            intermediate_ik_angle_tolerance_rad=bounded_angle,
        )
        adjusted.append((str(planner_id), str(backend)))

    if adjusted:
        print(
            "[P4-ROUTE-IK] TABLE BOWL intermediate approach tolerance: "
            f"{_TABLE_BOWL_INTERMEDIATE_POSITION_TOLERANCE_M:.3f} m / "
            f"{np.rad2deg(_TABLE_BOWL_INTERMEDIATE_ORIENTATION_TOLERANCE_RAD):.1f} deg",
            flush=True,
        )


def _patched_solve_points(
    self: CalibratedPickPlaceExecutor,
    ik: Any,
    points: list[np.ndarray],
    seed: np.ndarray,
    label: str,
    collision_checker: Any,
    allowed_environment_bodies: frozenset[int],
    target_rotation: np.ndarray | None = None,
):
    """Apply intermediate tolerance semantics to routed approach waypoints."""
    tolerance_label = label
    if label.startswith("Approaching above object route "):
        tolerance_label = "Approaching above object"
    return _ORIGINAL_SOLVE_POINTS(
        self,
        ik,
        points,
        seed,
        tolerance_label,
        collision_checker,
        allowed_environment_bodies,
        target_rotation,
    )


def _activate_initial_countertop_presentations(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
) -> None:
    model, data = dispatcher.scene.model, dispatcher.scene.data
    countertop_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "countertop"
    )
    if countertop_body < 0:
        return

    store = _presentation_store(dispatcher)
    for planner_id, row in dispatcher.inventory_by_id.items():
        context = row.get("source_context") or {}
        source_container = context.get("source_container")
        source_kind = str(context.get("source_kind"))
        if source_container in {"D1", "D2"}:
            continue
        if source_kind != "TABLE" and source_container not in {"C2", "B1", "C1"}:
            continue

        binding = dispatcher.binding_by_id.get(planner_id) or {}
        backend = binding.get("physical_backend_body")
        if not backend:
            continue
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, str(backend)
        )
        equality_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            _counter_weld_name(str(backend)),
        )
        if body_id < 0 or equality_id < 0:
            continue

        _set_weld_to_current_relative_pose(
            model, data, equality_id, countertop_body, body_id
        )
        data.eq_active[equality_id] = 1
        store[str(planner_id)] = {
            "backend_body": str(backend),
            "equality_id": int(equality_id),
        }

    if store:
        mujoco.mj_forward(model, data)
        print(
            "[P4-COUNTER-LOCK] authored countertop state preserved until first manipulation use",
            flush=True,
        )
        for planner_id, state in sorted(store.items()):
            print(
                f"  ACTIVE {planner_id}({state['backend_body']})",
                flush=True,
            )


def _release_countertop_presentation(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    *,
    reason: str,
) -> dict[str, Any] | None:
    state = _presentation_store(dispatcher).pop(object_id, None)
    if state is None:
        return None

    equality_id = int(state["equality_id"])
    if 0 <= equality_id < dispatcher.scene.model.neq:
        dispatcher.scene.data.eq_active[equality_id] = 0
    mujoco.mj_forward(dispatcher.scene.model, dispatcher.scene.data)

    print(
        f"[P4-COUNTER-LOCK] RELEASED {object_id}({state['backend_body']}) before {reason}",
        flush=True,
    )
    return {
        "released": True,
        "planner_id": object_id,
        "backend_body": state["backend_body"],
        "reason": reason,
        "direct_payload_pose_write": False,
        "direct_payload_velocity_write": False,
    }


def _patched_init(
    self: KitchenGroundTruthExecutionDispatcher,
    *args: Any,
    **kwargs: Any,
) -> None:
    _ORIGINAL_INIT(self, *args, **kwargs)
    _configure_table_bowl_intermediate_ik(self)
    _activate_initial_countertop_presentations(self)


def _patched_pick(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
) -> dict[str, Any]:
    released = _release_countertop_presentation(
        self, object_id, reason="PICK"
    )
    result = _ORIGINAL_PICK(self, object_id)
    if released is not None:
        result = {**result, "released_countertop_presentation": released}
    return result


def _patched_place(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
) -> dict[str, Any]:
    # A relative destination must become dynamic before the utensil/object is
    # inserted into it. Keeping it welded through the PLACE changes the
    # subsequent nested-object physics and caused the later bowl PICK failure.
    released_target = None
    if destination in self.binding_by_id:
        released_target = _release_countertop_presentation(
            self,
            destination,
            reason=f"RELATIVE PLACE TARGET FOR {object_id}",
        )

    result = _ORIGINAL_PLACE(self, object_id, destination)
    if released_target is not None:
        result = {
            **result,
            "released_relative_destination_presentation": released_target,
        }
    return result


def install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    scene_loader.build_scene_xml = _patched_build_scene_xml
    CalibratedPickPlaceExecutor._solve_points = _patched_solve_points
    KitchenGroundTruthExecutionDispatcher.__init__ = _patched_init
    KitchenGroundTruthExecutionDispatcher.pick = _patched_pick
    KitchenGroundTruthExecutionDispatcher.place = _patched_place
    _PATCHED = True
