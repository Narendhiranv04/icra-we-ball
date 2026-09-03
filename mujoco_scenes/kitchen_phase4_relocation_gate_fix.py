"""Phase-4 Kitchen fixes for relocated-object countertop placement.

The GT dispatcher uses ``allow_assisted_pick_recovery`` to authorize a bounded
post-contact PICK recovery.  That flag is PICK-specific, but the legacy PLACE
control flow also uses it as a gate: when it is True, a relocated object being
placed on the countertop skips the existing controlled-upright placement path
and later suppresses relocation recovery planning entirely.  The resulting
status is ``PLACEMENT_PLAN_NOT_FOUND`` with empty diagnostics even though no
placement plan was attempted.

For a relocated non-utensil payload, temporarily disable only that misplaced
PLACE gate while delegating to the existing physical dispatcher.  PICK recovery
semantics are restored immediately after the PLACE call.

The controlled-upright relocation primitive uses a fixed post-release settling
window.  A storage-retrieved vessel can be correctly released upright yet still
carry residual angular velocity when that fixed window expires.  If the only
reported failure is an unsettled linear/angular velocity after the payload has
already been physically released, continue ordinary MuJoCo dynamics for a
bounded interval and re-run the existing stable-placement validator.  No pose,
velocity, collision mask, planner action, or Phase-3 artifact is written or
changed by this recovery.
"""

from __future__ import annotations

from typing import Any

import mujoco

from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher


_ORIGINAL_PLACE = KitchenGroundTruthExecutionDispatcher.place
_PATCHED = False

_PASSIVE_SETTLE_MAX_STEPS = 6000
_PASSIVE_SETTLE_REQUIRED_STABLE_TICKS = 30


def _is_released_unsettled_failure(result: dict[str, Any]) -> bool:
    status = str(result.get("status", ""))
    return (
        status.startswith("RELEASED_PLACEMENT_OBJECT_UNSETTLED_ANG_VEL_")
        or status.startswith("RELEASED_PLACEMENT_OBJECT_UNSETTLED_LIN_VEL_")
    )


def _passively_settle_released_relocation(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
    original_result: dict[str, Any],
) -> dict[str, Any]:
    """Wait for a genuinely released relocation to satisfy existing checks."""
    low = self.phase_b.manipulation.executor
    if low.held_object is not None:
        return original_result

    stable_ticks = 0
    last_reason = str(original_result.get("status", "PLACEMENT_UNSETTLED"))
    last_telemetry = original_result.get("telemetry", {})

    print(
        "[P4-RELOCATION] payload physically released; extending passive "
        f"settle for {object_id}",
        flush=True,
    )

    for step in range(1, _PASSIVE_SETTLE_MAX_STEPS + 1):
        mujoco.mj_step(self.scene.model, self.scene.data)
        if self.step_callback is not None:
            self.step_callback(self.scene)

        valid, reason, telemetry = self.validate_stable_placement(
            object_id, destination
        )
        last_reason = reason
        last_telemetry = telemetry

        if valid:
            stable_ticks += 1
            if stable_ticks >= _PASSIVE_SETTLE_REQUIRED_STABLE_TICKS:
                self.update_object_to_countertop_location(object_id)
                print(
                    "[P4-RELOCATION] PASSIVE SETTLE VERIFIED "
                    f"{object_id} after {step} extra physics steps; "
                    f"lin={float(telemetry.get('linear_speed_mps', 0.0)):.3f} m/s, "
                    f"ang={float(telemetry.get('angular_speed_radps', 0.0)):.3f} rad/s",
                    flush=True,
                )
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
            # Passive settling is only allowed to resolve residual velocity.
            # If the object transitions to a different physical failure, stop
            # immediately and preserve that validator evidence.
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
        "post_release_passive_settle_max_steps": _PASSIVE_SETTLE_MAX_STEPS,
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
            if _is_released_unsettled_failure(result):
                result = _passively_settle_released_relocation(
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
