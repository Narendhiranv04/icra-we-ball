"""Phase-4 Kitchen fix for relocated-object countertop placement.

The GT dispatcher uses ``allow_assisted_pick_recovery`` to authorize a bounded
post-contact PICK recovery.  That flag is PICK-specific, but the legacy PLACE
control flow also uses it as a gate: when it is True, a relocated object being
placed on the countertop skips the existing controlled-upright placement path
and later suppresses relocation recovery planning entirely.  The resulting
status is ``PLACEMENT_PLAN_NOT_FOUND`` with empty diagnostics even though no
placement plan was attempted.

For a relocated non-utensil payload, temporarily disable only that misplaced
PLACE gate while delegating to the existing physical dispatcher.  PICK recovery
semantics are restored immediately after the PLACE call.  No object pose,
velocity, binding, Phase-3 artifact, collision mask, or planner action is
changed here.
"""

from __future__ import annotations

from typing import Any

from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher


_ORIGINAL_PLACE = KitchenGroundTruthExecutionDispatcher.place
_PATCHED = False


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
            return _ORIGINAL_PLACE(self, object_id, destination)
        finally:
            self.allow_assisted_pick_recovery = previous

    return _ORIGINAL_PLACE(self, object_id, destination)


def install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    KitchenGroundTruthExecutionDispatcher.place = _patched_place
    _PATCHED = True
