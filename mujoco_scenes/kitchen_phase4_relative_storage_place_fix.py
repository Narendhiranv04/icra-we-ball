"""Phase-4 Kitchen routing for object-relative PLACE into storage-resident targets.

The core GT PLACE dispatcher historically recenters every non-countertop-tool
PLACE at HOME before running the special soup-utensil insertion routine.  That
is correct when the destination bowl is already on the countertop, but wrong
when the grounded destination itself still lives inside an opened storage
region (for example K3's deep bowl in B1).  In that case the controller was
asking the HOME local-stance solver to reach into the box and its held-base
reposition stalled before the insertion even began.

This patch is execution-only.  For an object-relative destination whose live
source context is DRAWER/CUPBOARD/BOX, physically move the held payload to the
destination's existing required workspace first.  Then suppress exactly the
legacy one-shot HOME recenter request at the beginning of the delegated PLACE,
so the unchanged relative-placement controller plans from the correct storage
workspace.  Any later HOME motion is untouched.

No Phase-3 artifact, object pose/velocity, collision mask, functional binding,
or planner action is modified.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .kitchen_execution_entities import SourceKind
from .kitchen_execution_policy import KitchenWorkspace
from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher


_ORIGINAL_PLACE = KitchenGroundTruthExecutionDispatcher.place
_ORIGINAL_MOVE = KitchenGroundTruthExecutionDispatcher.move
_PATCHED = False

_STORAGE_SOURCE_KINDS = {
    SourceKind.DRAWER.value,
    SourceKind.CUPBOARD.value,
    SourceKind.BOX.value,
}


def _storage_relative_workspace(
    dispatcher: KitchenGroundTruthExecutionDispatcher,
    destination: str,
) -> tuple[KitchenWorkspace, str | None] | None:
    if destination not in dispatcher.binding_by_id:
        return None
    resolver = getattr(
        getattr(getattr(dispatcher, "phase_b", None), "manipulation", None),
        "placement_resolver",
        None,
    )
    if resolver is not None and destination in getattr(
        resolver, "serving_placements", {}
    ):
        return None
    row = dispatcher.inventory_by_id.get(destination) or {}
    if row.get("location") in {"serving_area", "countertop"}:
        return None
    context = row.get("source_context") or {}
    if str(context.get("source_kind")) not in _STORAGE_SOURCE_KINDS:
        return None
    required_raw = context.get("required_workspace")
    if required_raw is None:
        return None
    try:
        required = KitchenWorkspace(str(required_raw))
    except ValueError:
        return None
    container = context.get("source_container")
    return required, (None if container is None else str(container))


def _patched_move(
    self: KitchenGroundTruthExecutionDispatcher,
    workspace: KitchenWorkspace,
    *,
    carrying_object_id: str | None = None,
) -> dict[str, Any]:
    """Suppress only the delegated PLACE's legacy HOME recenter request."""
    pending = bool(
        getattr(self, "_phase4_relative_storage_home_redirect_pending", False)
    )
    override = getattr(
        self, "_phase4_relative_storage_required_workspace", None
    )
    override_object = getattr(
        self, "_phase4_relative_storage_held_object", None
    )
    requested = (
        workspace if isinstance(workspace, KitchenWorkspace)
        else KitchenWorkspace(str(workspace))
    )

    if (
        pending
        and override is not None
        and requested == KitchenWorkspace.HOME
        and carrying_object_id == override_object
    ):
        self._phase4_relative_storage_home_redirect_pending = False
        required = (
            override if isinstance(override, KitchenWorkspace)
            else KitchenWorkspace(str(override))
        )
        if self.current_workspace != required:
            return _ORIGINAL_MOVE(
                self, required, carrying_object_id=carrying_object_id
            )
        print(
            "[P4-RELATIVE-STORAGE] preserving destination workspace "
            f"{required.value} instead of legacy HOME recenter",
            flush=True,
        )
        return {
            "action": "MOVE",
            "arguments": [required.value],
            "success": True,
            "status": "RELATIVE_DESTINATION_WORKSPACE_PRESERVED",
            "requested_legacy_workspace": KitchenWorkspace.HOME.value,
            "effective_workspace": required.value,
            "carrying_object_id": carrying_object_id,
            "direct_payload_pose_write": False,
        }

    return _ORIGINAL_MOVE(
        self, requested, carrying_object_id=carrying_object_id
    )


def _patched_place(
    self: KitchenGroundTruthExecutionDispatcher,
    object_id: str,
    destination: str,
) -> dict[str, Any]:
    row = self.inventory_by_id.get(object_id) or {}
    source_context = row.get("source_context") or {}
    is_drawer_spoon = (
        str(source_context.get("source_kind")) == SourceKind.DRAWER.value
        or str(source_context.get("source_container")) in ("D1", "D2")
    )
    low = self.phase_b.manipulation.executor
    if is_drawer_spoon and self.current_workspace == KitchenWorkspace.HOME:
        if float(np.linalg.norm(self.scene.data.qpos[low.base_qpos][:2] - low.base_stance[:2])) > 0.02:
            print(
                f"[P4-RELATIVE-STORAGE] transitioning drawer-held utensil {object_id} "
                "base to canonical stance before placement",
                flush=True,
            )
            low.base_manipulation_target = low.base_stance.copy()
            for _ in range(900):
                low._command_base(low.base_manipulation_target)
                mujoco.mj_step(self.scene.model, self.scene.data)
                if self.step_callback is not None:
                    self.step_callback(self.scene)
                if low._base_at_target(low.base_manipulation_target):
                    break
            else:
                return {
                    "action": "PLACE",
                    "arguments": [object_id, destination],
                    "success": False,
                    "status": "DRAWER_CARRY_RETREAT_FAILED",
                    "message": "Carrying base retreat from drawer did not converge",
                    "direct_payload_pose_write": False,
                }
            low._restore_navigation_base_damping()
            mujoco.mj_forward(self.scene.model, self.scene.data)

    route = _storage_relative_workspace(self, destination)
    if route is None:
        return _ORIGINAL_PLACE(self, object_id, destination)

    required, container = route
    if container is not None and container not in self.physically_open_containers():
        return {
            "action": "PLACE",
            "arguments": [object_id, destination],
            "success": False,
            "status": "RELATIVE_DESTINATION_CONTAINER_CLOSED",
            "message": (
                f"Destination {destination} remains in closed storage {container}"
            ),
            "direct_payload_pose_write": False,
        }

    prep = None
    if self.current_workspace != required:
        print(
            "[P4-RELATIVE-STORAGE] moving held payload to destination workspace "
            f"{required.value} for PLACE({object_id}, {destination})",
            flush=True,
        )
        prep = _ORIGINAL_MOVE(
            self, required, carrying_object_id=object_id
        )
        if not prep.get("success", False):
            return {
                "action": "PLACE",
                "arguments": [object_id, destination],
                "success": False,
                "status": "RELATIVE_DESTINATION_MOVE_FAILED",
                "message": str(prep.get("status", "payload-aware MOVE failed")),
                "workspace_preparation": prep,
                "direct_payload_pose_write": False,
            }

    old_required = getattr(
        self, "_phase4_relative_storage_required_workspace", None
    )
    old_object = getattr(self, "_phase4_relative_storage_held_object", None)
    old_pending = getattr(
        self, "_phase4_relative_storage_home_redirect_pending", False
    )
    self._phase4_relative_storage_required_workspace = required
    self._phase4_relative_storage_held_object = object_id
    self._phase4_relative_storage_home_redirect_pending = True
    try:
        result = _ORIGINAL_PLACE(self, object_id, destination)
    finally:
        self._phase4_relative_storage_required_workspace = old_required
        self._phase4_relative_storage_held_object = old_object
        self._phase4_relative_storage_home_redirect_pending = old_pending

    if isinstance(result, dict):
        result = {
            **result,
            "relative_destination_workspace": required.value,
            "relative_destination_container": container,
            "workspace_preparation": prep,
            "legacy_home_recenter_suppressed": True,
        }
    return result


def install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    KitchenGroundTruthExecutionDispatcher.move = _patched_move
    KitchenGroundTruthExecutionDispatcher.place = _patched_place
    _PATCHED = True
