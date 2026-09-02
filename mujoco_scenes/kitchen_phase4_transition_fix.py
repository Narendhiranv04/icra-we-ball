"""Kitchen Phase-4 transition normalization after inspection/navigation.

Physical container inspection can leave the Google arm in the articulation pose
used at D1/D2/C1/C2/B1.  A later PICK may then start grasp-candidate ranking
from that live articulation posture even after the base has returned HOME.  The
Kitchen manipulation layer already provides ``_settle_navigation_posture`` to
physically fold the empty arm into the calibrated navigation configuration.
This module makes that transition mandatory before every physical PICK.

No robot or payload qpos/qvel is written here.  The arm is moved through its
position actuators and normal MuJoCo stepping before the existing PICK logic is
entered.  The frozen Phase-3 action sequence and object assignment are
unchanged.
"""

from __future__ import annotations

from typing import Any

from .kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher


_ORIGINAL_PICK = KitchenPhaseBExecutionDispatcher.pick
_PATCHED = False


def _patched_pick(
    self: KitchenPhaseBExecutionDispatcher,
    object_id: str,
) -> dict[str, Any]:
    """Physically normalize the empty arm before grasp planning."""
    preparation_steps = 0
    if self.manipulation.executor.held_object is None:
        preparation_steps = int(self.manipulation._settle_navigation_posture())

    result = _ORIGINAL_PICK(self, object_id)
    return {
        **result,
        "pre_pick_navigation_posture": {
            "performed": bool(preparation_steps),
            "physics_steps": preparation_steps,
            "method": "PHYSICAL_ARM_FOLD_BEFORE_PICK_PLANNING",
            "direct_robot_qpos_write": False,
            "direct_payload_state_write": False,
        },
    }


def install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    KitchenPhaseBExecutionDispatcher.pick = _patched_pick
    _PATCHED = True
