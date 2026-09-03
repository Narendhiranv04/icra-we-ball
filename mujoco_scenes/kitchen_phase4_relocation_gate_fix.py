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

A second legacy issue is the postcondition for a storage-retrieved cup/mug.
After a genuine controlled placement and release, the object can be visibly
upright and supported while the validator rejects one instantaneous angular
velocity norm. Angular speed alone is not the task-level PLACE postcondition.
For relocated VESSEL staging, an angular-only rejection is therefore followed
by a short observation horizon using ordinary MuJoCo dynamics. The placement is
accepted only if the vessel remains supported, upright, above the counter,
translation-stable, off the floor, and position-stable throughout that horizon.
Raw angular velocity (including roll/pitch/yaw components) is retained as
telemetry but is not itself the acceptance criterion. No pose/velocity write,
latch, collision-mask change, planner action, or Phase-3 artifact change is
used.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher


_ORIGINAL_PLACE = KitchenGroundTruthExecutionDispatcher.place
_PATCHED = False

# 0.5 s at the pinned MuJoCo 0.002 s timestep. This is long enough to expose a
# genuinely tipping/sliding placement without adding another multi-second fixed
# settle wait to every successful relocation.
_SUPPORT_HORIZON_STEPS = 250
_SUPPORT_REQUIRED_CONTACT_FRACTION = 0.90
_SUPPORT_MAX_XY_DRIFT_M = 0.015
_SUPPORT_MAX_TILT_DEG = 8.0
_SUPPORT_MAX_LINEAR_SPEED_MPS = 0.03
_SUPPORT_MIN_Z_M = 0.55

# If translation itself is still above the existing threshold, ordinary
# dynamics may settle it first. We never write qvel directly.
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


def _nonangular_snapshot_ok(telemetry: dict[str, Any]) -> bool:
    position = telemetry.get("position_xyz_m") or [0.0, 0.0, 0.0]
    return bool(
        telemetry.get("counter_contact", False)
        and not telemetry.get("floor_contact", False)
        and float(telemetry.get("tilt_deg", float("inf"))) <= _SUPPORT_MAX_TILT_DEG
        and float(telemetry.get("linear_speed_mps", float("inf"))) <= _SUPPORT_MAX_LINEAR_SPEED_MPS
        and float(position[2]) >= _SUPPORT_MIN_Z_M
    )


def _angular_components_world(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
) -> tuple[float, float, float, float]:
    binding = self.binding_by_id.get(object_id, {})
    backend = str(binding.get("physical_backend_body", object_id))
    body_id = mujoco.mj_name2id(
        self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
    )
    if body_id < 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        self.scene.model,
        self.scene.data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        0,
    )
    omega = np.asarray(velocity[:3], dtype=float)
    return (
        float(omega[0]),
        float(omega[1]),
        float(omega[2]),
        float(np.linalg.norm(omega)),
    )


def _accept_supported_vessel_horizon(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
    original_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Verify the achieved countertop PLACE over a short physical horizon."""
    if destination != "countertop" or not _released_angular_unsettled(original_result):
        return None

    binding = self.binding_by_id.get(object_id, {})
    if str(binding.get("grasp_family", "")) != "VESSEL":
        return None

    low = self.phase_b.manipulation.executor
    if low.held_object is not None:
        return None

    initial_telemetry = dict(original_result.get("telemetry") or {})
    if not initial_telemetry or not _nonangular_snapshot_ok(initial_telemetry):
        return None

    initial_position = np.asarray(
        initial_telemetry.get("position_xyz_m") or [0.0, 0.0, 0.0],
        dtype=float,
    )
    maximum_xy_drift = 0.0
    maximum_tilt = float(initial_telemetry.get("tilt_deg", 0.0))
    maximum_linear_speed = float(initial_telemetry.get("linear_speed_mps", 0.0))
    minimum_z = float(initial_position[2])
    contact_ticks = 1 if initial_telemetry.get("counter_contact", False) else 0
    sampled_ticks = 1
    last_telemetry = initial_telemetry
    omega_x, omega_y, omega_z, omega_norm = _angular_components_world(self, object_id)
    initial_omega = (omega_x, omega_y, omega_z, omega_norm)

    for _ in range(_SUPPORT_HORIZON_STEPS):
        mujoco.mj_step(self.scene.model, self.scene.data)
        if self.step_callback is not None:
            self.step_callback(self.scene)

        _, _, telemetry = self.validate_stable_placement(object_id, destination)
        last_telemetry = telemetry
        sampled_ticks += 1
        if telemetry.get("counter_contact", False):
            contact_ticks += 1

        position = np.asarray(
            telemetry.get("position_xyz_m") or [0.0, 0.0, 0.0],
            dtype=float,
        )
        maximum_xy_drift = max(
            maximum_xy_drift,
            float(np.linalg.norm(position[:2] - initial_position[:2])),
        )
        maximum_tilt = max(maximum_tilt, float(telemetry.get("tilt_deg", float("inf"))))
        maximum_linear_speed = max(
            maximum_linear_speed,
            float(telemetry.get("linear_speed_mps", float("inf"))),
        )
        minimum_z = min(minimum_z, float(position[2]))

        # Reject immediately if the object actually falls, tips, translates,
        # or drops below the support plane. Contact is handled as a fraction
        # because MuJoCo contact sets can flicker for isolated solver frames.
        if (
            telemetry.get("floor_contact", False)
            or maximum_tilt > _SUPPORT_MAX_TILT_DEG
            or maximum_linear_speed > _SUPPORT_MAX_LINEAR_SPEED_MPS
            or minimum_z < _SUPPORT_MIN_Z_M
            or maximum_xy_drift > _SUPPORT_MAX_XY_DRIFT_M
        ):
            return None

    contact_fraction = float(contact_ticks) / float(sampled_ticks)
    if contact_fraction < _SUPPORT_REQUIRED_CONTACT_FRACTION:
        return None

    final_omega = _angular_components_world(self, object_id)
    telemetry = dict(last_telemetry)
    telemetry.update({
        "placement_stability_mode": "SUPPORTED_POSE_HORIZON_V1",
        "support_horizon_steps": _SUPPORT_HORIZON_STEPS,
        "support_contact_fraction": contact_fraction,
        "maximum_xy_drift_m": maximum_xy_drift,
        "maximum_tilt_over_horizon_deg": maximum_tilt,
        "maximum_linear_speed_over_horizon_mps": maximum_linear_speed,
        "minimum_z_over_horizon_m": minimum_z,
        "initial_angular_velocity_world_radps": list(initial_omega[:3]),
        "initial_angular_speed_norm_radps": initial_omega[3],
        "final_angular_velocity_world_radps": list(final_omega[:3]),
        "final_angular_speed_norm_radps": final_omega[3],
        "legacy_total_angular_speed_rejection": original_result.get("status"),
    })
    self.update_object_to_countertop_location(object_id)

    backend = str(binding.get("physical_backend_body", object_id))
    print(
        "[P4-RELOCATION] SUPPORT-HORIZON STABILITY VERIFIED "
        f"{object_id}({backend}); contact={contact_fraction:.2f}, "
        f"xy_drift={maximum_xy_drift:.4f} m, max_tilt={maximum_tilt:.2f} deg, "
        f"max_lin={maximum_linear_speed:.3f} m/s",
        flush=True,
    )
    return {
        "action": "PLACE",
        "arguments": [object_id, destination],
        "success": True,
        "status": "PLACEMENT_COMPLETED_SUPPORTED_POSE_HORIZON",
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
            supported = _accept_supported_vessel_horizon(
                self, object_id, destination, synthetic
            )
            if supported is not None:
                return supported
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

            supported = _accept_supported_vessel_horizon(
                self, object_id, destination, result
            )
            if supported is not None:
                return supported

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
