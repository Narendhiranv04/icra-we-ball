"""Neutral adapter from baseline actions to the shared MuJoCo dispatcher."""

from __future__ import annotations

from collections.abc import Callable

from mujoco_scenes.tamp.physical_dispatcher import MuJoCoSkillDispatcher
from mujoco_scenes.tamp.skills import SkillAction, SkillStartError
from mujoco_scenes.tamp.state import ObservedState

from .models import Action, ActionResult, Entity, Observation, Region


def observation_from_state(
    scene: str, state: ObservedState, *, goal_satisfied: bool = False
) -> Observation:
    entities = tuple(
        Entity(item.object_id, "object", item.category, dict(item.facts))
        for item in state.objects.values()
        if item.visible
    )
    regions = tuple(
        Region(
            item.region_id,
            item.category,
            "unknown" if item.open is None else "open" if item.open else "closed",
            item.inspected,
        )
        for item in state.regions.values()
        if item.visible
    )
    return Observation(
        scene,
        state.revision,
        entities,
        regions,
        state.robot.as_dict(),
        goal_satisfied,
    )


class MuJoCoActionExecutor:
    """Run validated actions through the common physical skill dispatcher."""

    def __init__(
        self,
        dispatcher: MuJoCoSkillDispatcher,
        *,
        effect_sink: Callable[[tuple[str, ...]], None] | None = None,
        status_sink: Callable[[str], None] | None = None,
    ):
        self.dispatcher = dispatcher
        self.effect_sink = effect_sink
        self.status_sink = status_sink

    def prepare(self, actions: tuple[Action, ...]) -> ActionResult:
        result = self.dispatcher.prepare(
            tuple(SkillAction(action.skill, action.arguments) for action in actions)
        )
        if result.success:
            return ActionResult.succeeded()
        return ActionResult.failed(
            result.failure_code.value if result.failure_code else "execution_failed",
            result.message,
            recoverable=result.recoverable,
            details=result.details,
        )

    def execute(self, action: Action) -> ActionResult:
        if self.status_sink is not None:
            self.status_sink(f"Executing {action.skill}: {dict(action.arguments)}")
        try:
            self.dispatcher.start(SkillAction(action.skill, action.arguments))
        except SkillStartError as error:
            return ActionResult.failed(
                error.code.value,
                str(error),
                recoverable=error.recoverable,
            )
        except Exception as error:
            return ActionResult.failed(
                "internal_error",
                f"Physical dispatch raised {type(error).__name__}: {error}",
                recoverable=False,
            )
        try:
            result = self.dispatcher.update()
        except Exception as error:
            return ActionResult.failed(
                "internal_error",
                f"Physical update raised {type(error).__name__}: {error}",
                recoverable=False,
            )
        if result is None:
            return ActionResult.failed(
                "internal_error",
                "Physical dispatcher returned no terminal result.",
                recoverable=False,
            )
        if result.success:
            if self.effect_sink is not None:
                self.effect_sink(result.effects)
            if self.status_sink is not None:
                self.status_sink(f"Completed {action.skill}")
            return ActionResult.succeeded(*result.effects)
        if self.status_sink is not None:
            self.status_sink(f"Failed {action.skill}: {result.message}")
        return ActionResult.failed(
            result.failure_code.value if result.failure_code else "execution_failed",
            result.message,
            recoverable=result.recoverable,
            details=result.details,
        )
