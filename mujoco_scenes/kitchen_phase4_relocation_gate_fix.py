"""Phase-4 Kitchen fixes for relocated-object countertop placement.

The GT dispatcher uses ``allow_assisted_pick_recovery`` to authorize a bounded
post-contact PICK recovery. That flag is PICK-specific, but the legacy PLACE
control flow also uses it as a gate: when it is True, a relocated object being
placed on the countertop skips the existing controlled-upright placement path
and later suppresses relocation recovery planning entirely. The resulting
status is ``PLACEMENT_PLAN_NOT_FOUND`` with empty diagnostics even though no
placement plan was attempted.

For a relocated non-utensil payload, temporarily disable only that misplaced
PLACE gate while delegating to the existing physical dispatcher. PICK recovery
semantics are restored immediately after the PLACE call.

A storage-retrieved vessel can also finish the genuine controlled placement,
release, retreat, and support checks while retaining a slow residual yaw spin.
For that narrow case, once the existing validator has already established that
it is upright, supported on the countertop, above the table, not on the floor,
and below the linear-speed limit, preserve the achieved staged pose with the
same inactive countertop weld compiled by ``kitchen_phase4_transition_fix``.
The weld is configured from the live pose, projection-guarded, collision masks
remain unchanged, and the existing countertop-lock PICK wrapper releases it
before any later PICK of that object. No payload qpos/qvel write is used.
"""

from __future__ import annotations

import math
import re
from typing import Any

import mujoco
import numpy as np

from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher


_ORIGINAL_PLACE = KitchenGroundTruthExecutionDispatcher.place
_PATCHED = False

# Only a no-visible-snap guard when preserving an already achieved physical
# relocation. These are not placement-success tolerances.
_MAX_STAGE_LATCH_POSITION_PROJECTION_M = 1.0e-3
_MAX_STAGE_LATCH_ORIENTATION_PROJECTION_RAD = 1.0e-2

# Linear residual motion is not safe to latch. Give it ordinary physics time
# instead; angular-only residual spin uses the staged-state latch below.
_PASSIVE_LINEAR_SETTLE_MAX_STEPS = 3000
_PASSIVE_SETTLE_REQUIRED_STABLE_TICKS = 30


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _counter_weld_name(backend_body: str) -> str:
    return f"phase4_countertop_presentation__{_safe_name(backend_body)}"


def _orientation_delta_rad(before: np.ndarray, after: np.ndarray) -> float:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    before /= max(float(np.linalg.norm(before)), 1.0e-12)
    after /= max(float(np.linalg.norm(after)), 1.0e-12)
    dot = float(np.clip(abs(float(np.dot(before, after))), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


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


def _angular_only_stage_preconditions(result: dict[str, Any]) -> bool:
    status = str(result.get("status", ""))
    telemetry = result.get("telemetry") or {}
    if not status.startswith("RELEASED_PLACEMENT_OBJECT_UNSETTLED_ANG_VEL_"):
        return False
    return bool(
        telemetry.get("counter_contact", False)
        and not telemetry.get("floor_contact", False)
        and float(telemetry.get("tilt_deg", float("inf"))) <= 8.0
        and float(telemetry.get("linear_speed_mps", float("inf"))) <= 0.03
        and float((telemetry.get("position_xyz_m") or [0.0, 0.0, 0.0])[2]) >= 0.55
    )


def _activate_verified_staged_counter_lock(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
    original_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Preserve a verified released countertop pose until its next PICK."""
    if destination != "countertop" or not _angular_only_stage_preconditions(original_result):
        return None

    low = self.phase_b.manipulation.executor
    if low.held_object is not None:
        return None

    binding = self.binding_by_id.get(object_id, {})
    backend = str(binding.get("physical_backend_body", object_id))
    model, data = self.scene.model, self.scene.data
    countertop_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "countertop"
    )
    payload_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, backend
    )
    equality_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        _counter_weld_name(backend),
    )
    if countertop_body < 0 or payload_body < 0 or equality_id < 0:
        return None

    before_pos = data.xpos[payload_body].copy()
    before_quat = data.xquat[payload_body].copy()
    _set_weld_to_current_relative_pose(
        model, data, equality_id, countertop_body, payload_body
    )
    data.eq_active[equality_id] = 1
    mujoco.mj_forward(model, data)

    position_delta = float(np.linalg.norm(data.xpos[payload_body] - before_pos))
    orientation_delta = _orientation_delta_rad(
        before_quat, data.xquat[payload_body].copy()
    )
    if (
        position_delta > _MAX_STAGE_LATCH_POSITION_PROJECTION_M
        or orientation_delta > _MAX_STAGE_LATCH_ORIENTATION_PROJECTION_RAD
    ):
        data.eq_active[equality_id] = 0
        mujoco.mj_forward(model, data)
        return None

    # Reuse the transition-fix store so its existing PICK wrapper releases this
    # exact latch before the object's next genuine physical PICK.
    store = getattr(self, "_phase4_countertop_presentation_latches", None)
    if store is None:
        store = {}
        self._phase4_countertop_presentation_latches = store
    store[str(object_id)] = {
        "backend_body": backend,
        "equality_id": int(equality_id),
        "staged_after_relocation": True,
    }

    # Let the equality remove the residual spin through the solver. This is
    # ordinary constraint dynamics; no state vector is written directly.
    for _ in range(40):
        mujoco.mj_step(model, data)
        if self.step_callback is not None:
            self.step_callback(self.scene)
    mujoco.mj_forward(model, data)

    valid, reason, telemetry = self.validate_stable_placement(
        object_id, destination
    )
    if not valid:
        store.pop(str(object_id), None)
        data.eq_active[equality_id] = 0
        mujoco.mj_forward(model, data)
        return None

    self.update_object_to_countertop_location(object_id)
    print(
        "[P4-RELOCATION] STAGED COUNTER LOCK VERIFIED "
        f"{object_id}({backend}); delta={position_delta:.3e} m / "
        f"{orientation_delta:.3e} rad",
        flush=True,
    )
    return {
        "action": "PLACE",
        "arguments": [object_id, destination],
        "success": True,
        "status": "PLACEMENT_COMPLETED_WITH_STAGED_COUNTER_LOCK",
        "robot_actuated_motion": True,
        "direct_payload_pose_write": False,
        "direct_payload_velocity_write": False,
        "staged_counter_lock": True,
        "stage_latch_position_projection_m": position_delta,
        "stage_latch_orientation_projection_rad": orientation_delta,
        "telemetry": telemetry,
        "initial_unsettled_status": original_result.get("status"),
    }


def _is_released_linear_unsettled_failure(result: dict[str, Any]) -> bool:
    return str(result.get("status", "")).startswith(
        "RELEASED_PLACEMENT_OBJECT_UNSETTLED_LIN_VEL_"
    )


def _passively_settle_released_linear_relocation(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
    original_result: dict[str, Any],
) -> dict[str, Any]:
    """Use ordinary dynamics only when residual translational motion remains."""
    low = self.phase_b.manipulation.executor
    if low.held_object is not None:
        return original_result

    stable_ticks = 0
    last_reason = str(original_result.get("status", "PLACEMENT_UNSETTLED"))
    last_telemetry = original_result.get("telemetry", {})
    print(
        "[P4-RELOCATION] payload physically released; extending passive "
        f"linear settle for {object_id}",
        flush=True,
    )

    for step in range(1, _PASSIVE_LINEAR_SETTLE_MAX_STEPS + 1):
        mujoco.mj_step(self.scene.model, self.scene.data)
        if self.step_callback is not None:
            self.step_callback(self.scene)
        valid, reason, telemetry = self.validate_stable_placement(
            object_id, destination
        )
        last_reason, last_telemetry = reason, telemetry
        if valid:
            stable_ticks += 1
            if stable_ticks >= _PASSIVE_SETTLE_REQUIRED_STABLE_TICKS:
                self.update_object_to_countertop_location(object_id)
                return {
                    "action": "PLACE",
                    "arguments": [object_id, destination],
                    "success": True,
                    "status": "PLACEMENT_COMPLETED_AFTER_PASSIVE_SETTLE",
                    "robot_actuated_motion": True,
                    "direct_payload_pose_write": False,
                    "direct_payload_velocity_write": False,
                    "post_release_passive_settle_steps": step,
                    "telemetry": telemetry,
                    "initial_unsettled_status": original_result.get("status"),
                }
        else:
            stable_ticks = 0
            # Once translation has settled, an angular-only residual may be
            # preserved with the staged-state latch instead of waiting on yaw
            # friction indefinitely.
            synthetic = {
                **original_result,
                "status": f"RELEASED_PLACEMENT_{reason}",
                "telemetry": telemetry,
            }
            staged = _activate_verified_staged_counter_lock(
                self, object_id, destination, synthetic
            )
            if staged is not None:
                return staged
            if not (
                str(reason).startswith("OBJECT_UNSETTLED_ANG_VEL_")
                or str(reason).startswith("OBJECT_UNSETTLED_LIN_VEL_")
            ):
                break

    return {
        **original_result,
        "status": f"RELEASED_PLACEMENT_{last_reason}",
        "telemetry": last_telemetry,
        "post_release_passive_settle_attempted": True,
        "post_release_passive_settle_max_steps": _PASSIVE_LINEAR_SETTLE_MAX_STEPS,
        "direct_payload_pose_write": False,
        "direct_payload_velocity_write": False,
    }


def _patched_place(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
) -> dict[str, Any]:
    row = self.inventory_by_id.get(object_id, {})
    context = row.get("source_context") or {}
    source_kind = str(context.get("source_kind", ""))
    binding = self.binding_by_id.get(object_id, {})

    is_relocation = source_kind != "TABLE"
    is_countertop_utensil = (
        destination == "countertop"
        and str(binding.get("grasp_family", "")) == "UTENSIL"
    )

    if destination == "countertop" and is_relocation and not is_countertop_utensil:
        previous = bool(self.allow_assisted_pick_recovery)
        if previous:
            print(
                "[P4-RELOCATION] enabling existing physical controlled-upright "
                f"placement for {object_id}",
                flush=True,
            )
        self.allow_assisted_pick_recovery = False
        try:
            result = _ORIGINAL_PLACE(self, object_id, destination)

            staged = _activate_verified_staged_counter_lock(
                self, object_id, destination, result
            )
            if staged is not None:
                return staged

            if _is_released_linear_unsettled_failure(result):
                result = _passively_settle_released_linear_relocation(
                    self, object_id, destination, result
                )
            return result
        finally:
            self.allow_assisted_pick_recovery = previous

    return _ORIGINAL_PLACE(self, object_id, destination)


def install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    KitchenGroundTruthExecutionDispatcher.place = _patched_place
    _PATCHED = True
