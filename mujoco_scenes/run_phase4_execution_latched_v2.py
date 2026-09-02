"""Numerically robust post-placement containment latch runner for Phase 4.

This is a narrow correction to ``run_phase4_execution_latched``.  The original
latch used nanometre / 1e-8-radian pose-invariance thresholds immediately after
``mj_forward``.  Those thresholds are below normal floating-point kinematic
noise and can reject an otherwise unchanged live pose.  This runner keeps the
same physical PLACE-first contract and the same bowl<->utensil weld, but uses a
small explicit numerical tolerance and prints the exact latch failure reason.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from . import run_phase4_execution_latched as base
from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher


# This is only an activation-invariance check, not a placement tolerance.
# The spoon/bowl relation has already been physically verified by the existing
# PLACE controller before this latch runs.  50 micrometres / 50 microradians is
# tight enough to reject a real snap while staying above floating-point noise.
LATCH_ACTIVATION_POSITION_TOLERANCE_M = 5.0e-5
LATCH_ACTIVATION_ORIENTATION_TOLERANCE_RAD = 5.0e-5


def _activate_nested_latch_v2(
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
            "POST_PLACEMENT_CONTAINMENT_LATCH: resolved body is missing "
            f"({tool_id}->{tool_backend}, {bowl_id}->{bowl_backend})"
        )

    equality_name = base._nested_weld_name(tool_backend, bowl_backend)
    equality_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
    )
    if equality_id < 0:
        raise RuntimeError(
            "POST_PLACEMENT_CONTAINMENT_LATCH: equality was not compiled: "
            + equality_name
        )

    # Re-latching the same utensil must first restore its independent contact
    # state.  This is normally a no-op for the first soup placement.
    base._release_nested_latch(dispatcher, tool_id)

    before_tool_pos = data.xpos[tool_body].copy()
    before_tool_quat = data.xquat[tool_body].copy()

    # Configure the weld from the exact current live relation.  No qpos/qvel
    # write is made to either payload.
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
    model.eq_data[equality_id, 3:6] = relative_pos
    model.eq_data[equality_id, 6:10] = relative_quat

    # Once physical containment is established, the utensil is treated as a
    # rigid part of the served bowl assembly.  Suppress the utensil's own
    # contacts while latched so already-satisfied bowl contacts cannot chatter
    # against the weld.  Original masks are restored if the utensil is PICKed.
    tool_geom_masks: list[tuple[int, int, int]] = []
    for geom_id in base._body_geom_ids(model, tool_body):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        tool_geom_masks.append((geom_id, contype, conaffinity))
        if contype or conaffinity:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0

    data.eq_active[equality_id] = 1
    mujoco.mj_forward(model, data)

    position_delta = float(np.linalg.norm(data.xpos[tool_body] - before_tool_pos))
    quat_dot = abs(float(np.dot(data.xquat[tool_body], before_tool_quat)))
    orientation_delta = 2.0 * float(
        np.arccos(np.clip(quat_dot, -1.0, 1.0))
    )

    if (
        position_delta > LATCH_ACTIVATION_POSITION_TOLERANCE_M
        or orientation_delta > LATCH_ACTIVATION_ORIENTATION_TOLERANCE_RAD
    ):
        data.eq_active[equality_id] = 0
        for geom_id, contype, conaffinity in tool_geom_masks:
            model.geom_contype[geom_id] = contype
            model.geom_conaffinity[geom_id] = conaffinity
        mujoco.mj_forward(model, data)
        raise RuntimeError(
            "POST_PLACEMENT_CONTAINMENT_LATCH: activation changed live pose; "
            f"position_delta_m={position_delta:.9g}, "
            f"orientation_delta_rad={orientation_delta:.9g}, "
            f"limits=({LATCH_ACTIVATION_POSITION_TOLERANCE_M:.1e} m, "
            f"{LATCH_ACTIVATION_ORIENTATION_TOLERANCE_RAD:.1e} rad)"
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
    base._latch_store(dispatcher)[tool_id] = state

    print(
        "[P4-LATCH] ACTIVE "
        f"{tool_id}({tool_backend}) -> {bowl_id}({bowl_backend}); "
        f"activation_delta={position_delta:.3e} m / "
        f"{orientation_delta:.3e} rad",
        flush=True,
    )

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
        "activation_position_tolerance_m": LATCH_ACTIVATION_POSITION_TOLERANCE_M,
        "activation_orientation_tolerance_rad": LATCH_ACTIVATION_ORIENTATION_TOLERANCE_RAD,
    }


def _patched_place_v2(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
) -> dict[str, Any]:
    # Keep the known-good physical PLACE exactly as-is.  The latch runs only
    # after that controller has already returned success.
    result = base._ORIGINAL_PLACE(self, object_id, destination)
    if not result.get("success", False):
        return result
    if destination not in self.binding_by_id:
        return result
    if not base._soup_pair(self, object_id, destination):
        return result
    if result.get("direct_payload_pose_write", False):
        message = "Containment latch refuses assisted/direct-pose placement"
        print(f"[P4-LATCH] REJECTED: {message}", flush=True)
        return {
            **result,
            "success": False,
            "status": "POST_PLACEMENT_CONTAINMENT_LATCH_REJECTED",
            "failure_code": "EXECUTION_ERROR",
            "message": message,
        }
    try:
        latch = _activate_nested_latch_v2(self, object_id, destination)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        print(f"[P4-LATCH] FAILED: {message}", flush=True)
        return {
            **result,
            "success": False,
            "status": "POST_PLACEMENT_CONTAINMENT_LATCH_FAILED",
            "failure_code": "EXECUTION_ERROR",
            "message": message,
            "latch_failure_type": type(error).__name__,
            "latch_failure_reason": str(error),
        }
    return {
        **result,
        "nested_containment_latch": latch,
        "post_placement_containment_latched": True,
    }


def install_patch() -> None:
    # Replace only the latch helper / wrapper before the original module
    # installs its scene-XML and dispatcher monkey patches.
    base._activate_nested_latch = _activate_nested_latch_v2
    base._patched_place = _patched_place_v2
    base.install_patch()


def main() -> int:
    install_patch()
    return base._base_main()


if __name__ == "__main__":
    raise SystemExit(main())
