"""Physical Google Robot dispatcher for kitchen container actions.

This is the Phase-A boundary: symbolic OPEN/CLOSE requests are refined with a
mobile-base move when required, then delegated to contact-gated articulation.
It never calls the direct perception-only container actuator helpers.
"""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

import mujoco

from .kitchen_articulation import GoogleKitchenArticulationExecutor
from .kitchen_execution_policy import (
    KitchenExecutionPolicy,
    KitchenWorkspace,
    WORKSPACE_DESTINATIONS,
)
from .mobile_motion import MobileMoveExecutor


PHYSICAL_TO_WORKSPACE = {
    "home": KitchenWorkspace.HOME,
    "cupboard1": KitchenWorkspace.LEFT_SIDE,
    "right_side": KitchenWorkspace.RIGHT_SIDE,
}


class KitchenGoogleExecutionDispatcher:
    """Synchronously refine and execute MOVE + OPEN/CLOSE requests."""

    def __init__(self, scene, *, held_object_getter=None, step_callback=None):
        if scene.robot_name != "google":
            raise ValueError("Kitchen physical execution requires --robot google")
        self.scene = scene
        self.step_callback = step_callback
        self.navigation = MobileMoveExecutor(scene.model, scene.data, "google")
        self.articulation = GoogleKitchenArticulationExecutor(
            scene, held_object_getter=held_object_getter,
            step_callback=step_callback,
        )
        self.policy = KitchenExecutionPolicy()

    @property
    def current_workspace(self) -> KitchenWorkspace:
        return PHYSICAL_TO_WORKSPACE[self.navigation.current_physical_location]

    def _move(self, workspace: KitchenWorkspace, *, max_steps: int = 180_000) -> dict[str, Any]:
        destination = WORKSPACE_DESTINATIONS[workspace]
        start = self.navigation.current_physical_location
        started = time.perf_counter()
        self.navigation.request_move(destination)
        physics_steps = 0
        while self.navigation.busy and physics_steps < max_steps:
            self.navigation.update()
            mujoco.mj_step(self.scene.model, self.scene.data)
            if self.step_callback is not None:
                self.step_callback()
            physics_steps += 1
        success = not self.navigation.busy
        return {
            "action": "MOVE",
            "source_workspace": PHYSICAL_TO_WORKSPACE[start].value,
            "target_workspace": workspace.value,
            "destination": destination,
            "success": success,
            "status": self.navigation.status,
            "physics_steps": physics_steps,
            "duration_s": time.perf_counter() - started,
        }

    def request(self, action: str, container: str, *, execute: bool) -> dict[str, Any]:
        refinement = self.policy.refine(action, container, self.current_workspace)
        record: dict[str, Any] = {
            "request": {"action": action.upper(), "target": container},
            "refinement": {
                **asdict(refinement),
                "requested_action": refinement.requested_action.value,
                "required_workspace": refinement.required_workspace.value,
                "starting_workspace": refinement.starting_workspace.value,
            },
            "execution_enabled": bool(execute),
            "steps": [],
        }
        if not execute:
            record.update(status="PLAN_ONLY", success=True)
            return record
        if refinement.auto_move_inserted:
            move = self._move(refinement.required_workspace)
            move["automatically_inserted"] = True
            record["steps"].append(move)
            if not move["success"]:
                record.update(status="NAVIGATION_FAILED", success=False)
                return record
        segment_targets = [None]
        segment_results = []
        for segment_index, target_q in enumerate(segment_targets):
            result = self.articulation.execute(
                action, container, self.current_workspace,
                target_q_override=target_q,
            )
            result_dict = result.to_dict()
            result_dict["articulation_segment"] = segment_index + 1
            result_dict["articulation_segment_count"] = len(segment_targets)
            record["steps"].append(result_dict)
            segment_results.append(result)
            if not result.success:
                break
        result = segment_results[-1]
        record["success"] = all(item.success for item in segment_results)
        record["status"] = (
            "EXECUTION_SUCCESS" if record["success"] else result.status
        )
        if record["success"] and action.upper() == "OPEN":
            # State bookkeeping occurs only after the measured physical
            # postcondition succeeds; it never drives the joint.
            self.scene.record_container_opened(container)
        elif record["success"] and action.upper() == "CLOSE":
            self.scene.state.container_open_state[container] = False
        return record
