"""Physical Living Room execution behind the shared baseline executor contract.

The calibrated Living Room skills already live in
``mujoco_scenes.living_room_discovery_runtime``: they command the mobile base,
run the pick/place controllers, verify the physical ON relation, and update the
observable state.  This module only adapts them to the ``Action``/``ActionResult``
contract the comparison baselines consume, so every method executes through the
same controllers rather than a per-baseline copy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from .models import Action, ActionResult

# The skills report domain codes; the shared failure vocabulary is what the
# baselines feed back to a model and record in their artifacts.
FAILURE_CODES: Mapping[str, str] = {
    "GRIPPER_OCCUPIED": "hand_not_empty",
    "NOT_HELD": "precondition_failed",
    "GRASP": "grasp_failed",
    "PLACE": "placement_failed",
    "UNSUPPORTED": "unsupported_subgoal",
    "EXECUTION_FAILED": "execution_failed",
}


class LivingRoomPhysicalExecutor:
    """Run PICK/PLACE through the calibrated Google-robot Living Room skills."""

    def __init__(
        self,
        runtime: Any,
        *,
        effect_sink: Callable[[tuple[str, ...]], None] | None = None,
        status_sink: Callable[[str], None] | None = None,
    ):
        for required in ("execute_phase2_action", "observe_state"):
            if not hasattr(runtime, required):
                raise TypeError(
                    f"Living Room physical runtime must provide {required}()"
                )
        self.runtime = runtime
        self.effect_sink = effect_sink
        self.status_sink = status_sink
        self.executed_actions = 0

    @staticmethod
    def _arguments(action: Action) -> list[str] | None:
        skill = action.skill.upper()
        values = action.arguments
        if skill == "PICK":
            object_id = values.get("object_id")
            return [str(object_id)] if object_id else None
        if skill == "PLACE":
            object_id = values.get("object_id")
            region_id = values.get("region_id") or values.get("destination")
            return [str(object_id), str(region_id)] if object_id and region_id else None
        return None

    def execute(self, action: Action) -> ActionResult:
        skill = action.skill.upper()
        arguments = self._arguments(action)
        if arguments is None:
            return ActionResult.failed(
                "unsupported_subgoal",
                f"Living Room physical execution does not implement {action.skill}",
                recoverable=False,
            )
        if self.status_sink is not None:
            self.status_sink(f"Executing {skill}({', '.join(arguments)})")
        outcome = self.runtime.execute_phase2_action(
            {"action": skill, "arguments": arguments}
        )
        self.executed_actions += 1
        effects = tuple(str(item) for item in (outcome.get("effects") or ()))
        if outcome.get("success"):
            if self.effect_sink is not None:
                self.effect_sink(effects)
            if self.status_sink is not None:
                self.status_sink(f"Completed {skill}")
            return ActionResult.succeeded(*effects)
        raw_code = str(outcome.get("failure_code") or "EXECUTION_FAILED")
        message = str(outcome.get("message") or "").strip()
        if self.status_sink is not None:
            self.status_sink(f"Failed {skill}: {raw_code}")
        return ActionResult.failed(
            FAILURE_CODES.get(raw_code, "execution_failed"),
            message or f"Living Room skill {skill} reported {raw_code}",
            # A failed skill leaves the observable state intact, so the
            # executive may re-observe and replan rather than abort.
            recoverable=True,
            details={"skill_failure_code": raw_code},
        )


def build_living_room_physical_runtime(
    variant: str,
    output_dir: Any,
    *,
    camera_count: int = 3,
    show_viewer: bool = False,
    viewer_camera: str = "free",
    image_width: int = 960,
    image_height: int = 540,
):
    """Construct the calibrated physical Living Room runtime.

    ``image_width``/``image_height`` matter to any method whose grounding reads
    the rendered frames: retrieval crops them for CLIP, so a physical episode
    rendered at a different size than its planning episode would score
    different crops and could execute a different assignment than it reported.
    """
    from mujoco_scenes.living_room_discovery_runtime import LivingRoomDiscoveryRuntime

    return LivingRoomDiscoveryRuntime(
        variant,
        output_dir,
        camera_count=camera_count,
        show_viewer=show_viewer,
        viewer_camera=viewer_camera,
        image_width=image_width,
        image_height=image_height,
    )
