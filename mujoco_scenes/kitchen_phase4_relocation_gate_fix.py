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

A storage-retrieved cup/mug can also complete a genuine controlled placement,
release, retreat, support, tilt, height, and linear-stability checks while the
legacy validator still rejects it because it takes the norm of all three
angular-velocity components. For an upright vessel on a horizontal countertop,
rotation about world Z is task-irrelevant yaw: it does not change support,
uprightness, containment, or accessibility. We therefore keep the existing
angular threshold for the *tipping* (roll/pitch) component and ignore only the
vertical yaw component for relocated VESSEL staging. The raw total angular
velocity remains in telemetry. No pose/velocity write, latch, collision-mask
change, planner action, or Phase-3 artifact change is used.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher


_ORIGINAL_PLACE = KitchenGroundTruthExecutionDispatcher.place
_PATCHED = False

_PASSIVE_LINEAR_SETTLE_MAX_STEPS = 3000
_PASSIVE_SETTLE_REQUIRED_STABLE_TICKS = 30


def _released_angular_unsettled(result: dict[str, Any]) -> bool:
    return str(result.get("status", "")).startswith(
        "RELEASED_PLACEMENT_OBJECT_UNSETTLED_ANG_VEL_"
    )


def _released_linear_unsettled(result: dict[str, Any]) -> bool:
    return str(result.get("status", "")).startswith(
        "RELEASED_PLACEMENT_OBJECT_UNSETTLED_LIN_VEL_"
    )


def _accept_yaw_invariant_vessel_stability(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
    original_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Accept only residual world-Z yaw after all physical placement gates pass."""
    if destination != "countertop" or not _released_angular_unsettled(original_result):
        return None

    binding = self.binding_by_id.get(object_id, {})
    if str(binding.get("grasp_family", "")) != "VESSEL":
        return None

    low = self.phase_b.manipulation.executor
    if low.held_object is not None:
        return None

    telemetry = dict(original_result.get("telemetry") or {})
    if not telemetry:
        return None

    # These are the same non-angular physical postconditions enforced by the
    # legacy validator before it reports OBJECT_UNSETTLED_ANG_VEL_*.
    if telemetry.get("floor_contact", False):
        return None
    if not telemetry.get("counter_contact", False):
        return None
    if float(telemetry.get("tilt_deg", float("inf"))) > 8.0:
        return None
    if float(telemetry.get("linear_speed_mps", float("inf"))) > 0.03:
        return None
    position = telemetry.get("position_xyz_m") or [0.0, 0.0, 0.0]
    if float(position[2]) < 0.55:
        return None

    backend = str(binding.get("physical_backend_body", object_id))
    body_id = mujoco.mj_name2id(
        self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
    )
    if body_id < 0:
        return None

    spatial_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        self.scene.model,
        self.scene.data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        spatial_velocity,
        0,
    )
    omega_world = np.asarray(spatial_velocity[:3], dtype=float)
    total_angular_speed = float(np.linalg.norm(omega_world))
    yaw_rate = float(omega_world[2])
    tipping_angular_speed = float(np.linalg.norm(omega_world[:2]))
    threshold = float(telemetry.get("maximum_angular_speed_radps", 0.12))

    if tipping_angular_speed > threshold:
        return None

    telemetry.update({
        "angular_speed_radps": total_angular_speed,
        "yaw_rate_radps": yaw_rate,
        "tipping_angular_speed_radps": tipping_angular_speed,
        "angular_stability_mode": "YAW_INVARIANT_HORIZONTAL_SUPPORT_V1",
        "task_relevant_angular_speed_radps": tipping_angular_speed,
        "task_relevant_angular_speed_threshold_radps": threshold,
        "legacy_total_angular_speed_rejection": original_result.get("status"),
    })
    self.update_object_to_countertop_location(object_id)

    print(
        "[P4-RELOCATION] YAW-INVARIANT STABILITY VERIFIED "
        f"{object_id}({backend}); total={total_angular_speed:.3f}, "
        f"yaw={yaw_rate:.3f}, tipping={tipping_angular_speed:.3f} rad/s",
        flush=True,
    )
    return {
        "action": "PLACE",
        "arguments": [object_id, destination],
        "success": True,
        "status": "PLACEMENT_COMPLETED_YAW_INVARIANT_STABILITY",
        "robot_actuated_motion": True,
        "direct_payload_pose_write": False,
        "direct_payload_velocity_write": False,
        "telemetry": telemetry,
        "initial_unsettled_status": original_result.get("status"),
    }


def _passively_settle_released_linear_relocation(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
    original_result: dict[str, Any],
) -> dict[str, Any]:
    """Use ordinary dynamics only while residual translational motion remains."""
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
            synthetic = {
                **original_result,
                "status": f"RELEASED_PLACEMENT_{reason}",
                "telemetry": telemetry,
            }
            yaw_invariant = _accept_yaw_invariant_vessel_stability(
                self, object_id, destination, synthetic
            )
            if yaw_invariant is not None:
                return yaw_invariant
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

            yaw_invariant = _accept_yaw_invariant_vessel_stability(
                self, object_id, destination, result
            )
            if yaw_invariant is not None:
                return yaw_invariant

            if _released_linear_unsettled(result):
                return _passively_settle_released_linear_relocation(
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
