"""Observation-bounded discovery and failure replanning.

This module is the MuJoCo counterpart of the earlier Robust-TAMP execution
loop.  A planner proposes *task-level* skills from the current observation;
continuous motion remains the responsibility of the existing MuJoCo skill
dispatcher.  The planner is called again only after a recoverable execution
failure, a genuinely newly visible object, or an incomplete final goal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .events import EventLog
from .skills import FailureCode, SkillAction, SkillDispatcher, SkillResult, SkillStartError
from .state import ObservedState


class PlanStatus(StrEnum):
    PLAN = "PLAN"
    GOAL_COMPLETE = "GOAL_COMPLETE"
    NO_VALID_PLAN = "NO_VALID_PLAN"


class RecoverablePlanningError(RuntimeError):
    """A model response can be retried from the current observation."""


@dataclass(frozen=True)
class CameraObservation:
    """A planner-facing camera frame encoded as an image data URL."""

    camera: str
    data_url: str

    def as_dict(self) -> dict[str, str]:
        return {"camera": self.camera, "data_url": self.data_url}


@dataclass(frozen=True)
class PlanningSnapshot:
    """One fresh bounded state and the RGB views captured with it."""

    state: ObservedState
    images: tuple[CameraObservation, ...] = ()


@dataclass(frozen=True)
class ReplanEvent:
    """The only information that causes an existing plan to be replaced."""

    event_type: str
    event_id: str
    message: str
    action: SkillAction | None = None
    evidence: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "message": self.message,
            "action": (
                {"name": self.action.name, "arguments": dict(self.action.arguments)}
                if self.action is not None
                else None
            ),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class PlannerRequest:
    scene: str
    goal: str
    snapshot: PlanningSnapshot
    completed_actions: tuple[SkillAction, ...]
    remaining_actions: tuple[SkillAction, ...]
    replan_event: ReplanEvent | None = None


@dataclass(frozen=True)
class PlannerResult:
    status: PlanStatus
    actions: tuple[SkillAction, ...] = ()
    raw_output: str = ""
    latency_s: float | None = None
    message: str = ""


class ReplanningPlanner(Protocol):
    def plan(self, request: PlannerRequest) -> PlannerResult:
        """Return a direct task-level plan from the current observation."""


SnapshotObserver = Callable[[], PlanningSnapshot | ObservedState]
GoalVerifier = Callable[[str, ObservedState, Sequence[Mapping[str, object]]], bool]
DiscoveryFilter = Callable[[frozenset[str], ObservedState], frozenset[str]]
EffectSink = Callable[[tuple[str, ...]], None]
PreActionCheck = Callable[[SkillAction, ObservedState], SkillResult | None]


def _as_snapshot(value: PlanningSnapshot | ObservedState) -> PlanningSnapshot:
    if isinstance(value, PlanningSnapshot):
        return value
    if isinstance(value, ObservedState):
        return PlanningSnapshot(value)
    raise TypeError("Snapshot observer must return PlanningSnapshot or ObservedState")


def _action_record(action: SkillAction, result: SkillResult) -> dict[str, object]:
    return {
        "action": {"name": action.name, "arguments": dict(action.arguments)},
        "success": result.success,
        "effects": list(result.effects),
        "failure_code": result.failure_code.value if result.failure_code else None,
        "message": result.message,
        "details": dict(result.details),
    }


class DiscoveryReplanningExecutive:
    """Execute direct skills while replanning from newly observed state.

    Initial visibility is recorded once.  An object can therefore disappear
    behind another object without being rediscovered.  Only a first appearance
    after the episode begins triggers a discovery event.
    """

    TERMINAL_MODES = frozenset({"idle", "complete", "failed"})

    def __init__(
        self,
        *,
        scene: str,
        goal: str,
        observer: SnapshotObserver,
        planner: ReplanningPlanner,
        dispatcher: SkillDispatcher,
        goal_verifier: GoalVerifier,
        discovery_filter: DiscoveryFilter | None = None,
        effect_sink: EffectSink | None = None,
        pre_action_check: PreActionCheck | None = None,
        max_replans: int = 5,
        max_model_calls: int | None = None,
        max_actions: int = 80,
        defer_discovery_while_holding: bool = True,
        event_log: EventLog | None = None,
    ):
        if not scene.strip() or not goal.strip():
            raise ValueError("scene and goal must be non-empty")
        if isinstance(max_replans, bool) or not isinstance(max_replans, int) or max_replans < 0:
            raise ValueError("max_replans must be a non-negative integer")
        if (
            max_model_calls is not None
            and (
                isinstance(max_model_calls, bool)
                or not isinstance(max_model_calls, int)
                or max_model_calls <= 0
            )
        ):
            raise ValueError("max_model_calls must be a positive integer or None")
        if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions <= 0:
            raise ValueError("max_actions must be a positive integer")
        self.scene = scene
        self.goal = goal
        self.observer = observer
        self.planner = planner
        self.dispatcher = dispatcher
        self.goal_verifier = goal_verifier
        self.discovery_filter = discovery_filter
        self.effect_sink = effect_sink
        self.pre_action_check = pre_action_check
        self.max_replans = max_replans
        self.max_model_calls = max_model_calls
        self.max_actions = max_actions
        self.defer_discovery_while_holding = bool(defer_discovery_while_holding)
        self.events = event_log or EventLog()

        self.mode = "idle"
        self.status = "Discovery replanning idle"
        self.snapshot: PlanningSnapshot | None = None
        self.history: list[dict[str, object]] = []
        self.completed_actions: list[SkillAction] = []
        self._known_visible_objects: set[str] = set()
        self._actions: tuple[SkillAction, ...] = ()
        self._action_index = 0
        self._active = False
        self._executed_actions = 0
        self._replans = 0
        self._model_calls = 0
        self._planning_latency_s = 0.0
        self._pending_discovery: set[str] = set()
        self.last_event: ReplanEvent | None = None

    @property
    def busy(self) -> bool:
        return self.mode not in self.TERMINAL_MODES

    @property
    def replans(self) -> int:
        return self._replans

    @property
    def executed_actions(self) -> int:
        return self._executed_actions

    @property
    def model_calls(self) -> int:
        return self._model_calls

    @property
    def planning_latency_s(self) -> float:
        return self._planning_latency_s

    @property
    def remaining_actions(self) -> tuple[SkillAction, ...]:
        return self._actions[self._action_index :]

    def start(self) -> None:
        if self.busy:
            raise RuntimeError("Discovery replanning is already running")
        self.mode = "planning"
        self.status = "Capturing initial observation"
        self.history.clear()
        self.completed_actions.clear()
        self._known_visible_objects.clear()
        self._actions = ()
        self._action_index = 0
        self._active = False
        self._executed_actions = 0
        self._replans = 0
        self._model_calls = 0
        self._planning_latency_s = 0.0
        self._pending_discovery.clear()
        self.last_event = None
        if not self._observe("initial_observation"):
            return
        assert self.snapshot is not None
        self._known_visible_objects.update(
            object_id
            for object_id, observation in self.snapshot.state.objects.items()
            if observation.visible
        )
        self.events.append(
            "discovery_episode_started",
            scene=self.scene,
            goal=self.goal,
            observed_revision=self.snapshot.state.revision,
            initial_visible_objects=sorted(self._known_visible_objects),
        )
        self._request_plan(None)

    def update(self) -> None:
        if self.mode != "executing":
            return
        if self._executed_actions >= self.max_actions:
            self._fail(FailureCode.PRECONDITION_FAILED, "Execution action budget exhausted")
            return
        action = self._actions[self._action_index]
        if not self._active:
            self.status = f"Executing {action.name} ({self._action_index + 1}/{len(self._actions)})"
            if self.pre_action_check is not None:
                assert self.snapshot is not None
                try:
                    precheck = self.pre_action_check(action, self.snapshot.state)
                except Exception as error:
                    precheck = SkillResult.failed(
                        FailureCode.INTERNAL_ERROR,
                        f"Observed-state precheck failed: {error}",
                        recoverable=False,
                    )
                if precheck is not None and not precheck.success:
                    self._record(action, precheck)
                    self._recover_or_fail(
                        action,
                        self.history[-1],
                        recoverable=precheck.recoverable,
                    )
                    return
            try:
                self.dispatcher.start(action)
            except SkillStartError as error:
                self._record(action, SkillResult.failed(error.code, str(error), recoverable=error.recoverable))
                self._recover_or_fail(action, self.history[-1], recoverable=error.recoverable)
                return
            except Exception as error:
                self._record(action, SkillResult.failed(FailureCode.INTERNAL_ERROR, str(error), recoverable=False))
                self._recover_or_fail(action, self.history[-1], recoverable=False)
                return
            self._active = True
            self.events.append("discovery_skill_started", action=action.name, arguments=dict(action.arguments))

        try:
            result = self.dispatcher.update()
        except Exception as error:
            result = SkillResult.failed(FailureCode.INTERNAL_ERROR, str(error), recoverable=False)
        if result is None:
            return
        self._active = False
        self._executed_actions += 1
        self._record(action, result)
        if result.success and self.effect_sink is not None:
            try:
                self.effect_sink(result.effects)
            except Exception as error:
                self._fail(
                    FailureCode.INTERNAL_ERROR,
                    f"Effect recording failed: {error}",
                )
                return
        if not self._observe("post_action_observation"):
            return
        if not result.success:
            self._recover_or_fail(action, self.history[-1], recoverable=result.recoverable)
            return

        self.completed_actions.append(action)
        self._action_index += 1
        discovered = self._newly_visible_objects()
        if discovered:
            self._pending_discovery.update(discovered)
            self.events.append(
                "objects_discovered",
                after_action=action.name,
                object_ids=sorted(discovered),
                held_object=self.snapshot.state.robot.held_object if self.snapshot else None,
            )
        if self._pending_discovery and self._can_interrupt_for_discovery():
            self._replan_for_discovery(action)
            return
        if self._action_index >= len(self._actions):
            self._finish_plan()

    def _observe(self, event: str) -> bool:
        try:
            self.snapshot = _as_snapshot(self.observer())
            return True
        except Exception as error:
            self._fail(FailureCode.INTERNAL_ERROR, f"Observation failed: {error}")
            self.events.append("discovery_observation_failed", event_name=event, message=str(error))
            return False

    def _newly_visible_objects(self) -> frozenset[str]:
        if self.snapshot is None:
            return frozenset()
        current = {
            object_id
            for object_id, observation in self.snapshot.state.objects.items()
            if observation.visible
        }
        new = current - self._known_visible_objects
        self._known_visible_objects.update(current)
        if self.discovery_filter is not None:
            return self.discovery_filter(frozenset(new), self.snapshot.state)
        return frozenset(new)

    def _can_interrupt_for_discovery(self) -> bool:
        if self.snapshot is None:
            return False
        return not (
            self.defer_discovery_while_holding
            and self.snapshot.state.robot.held_object is not None
        )

    def _replan_for_discovery(self, action: SkillAction) -> None:
        discovered = sorted(self._pending_discovery)
        self._pending_discovery.clear()
        event = ReplanEvent(
            "discovery",
            "new_object_discovered",
            "New object observations require a plan from the current checkpoint.",
            action,
            {"newly_visible_object_ids": discovered},
        )
        self.events.append("discovery_replan_requested", **event.as_dict())
        self._begin_replan(event)

    def _recover_or_fail(self, action: SkillAction, record: Mapping[str, object], *, recoverable: bool) -> None:
        code_text = str(record.get("failure_code") or FailureCode.INTERNAL_ERROR.value)
        try:
            code = FailureCode(code_text)
        except ValueError:
            code = FailureCode.INTERNAL_ERROR
        message = str(record.get("message") or "Physical skill failed")
        if not recoverable:
            self._fail(code, message)
            return
        event = ReplanEvent(
            "failure",
            code.value,
            message,
            action,
            {"skill_result": dict(record)},
        )
        self.events.append("failure_replan_requested", **event.as_dict())
        self._begin_replan(event)

    def _finish_plan(self) -> None:
        if self.snapshot is None:
            self._fail(FailureCode.INTERNAL_ERROR, "No observation is available for final verification")
            return
        try:
            complete = self.goal_verifier(self.goal, self.snapshot.state, self.history)
        except Exception as error:
            self._fail(FailureCode.INTERNAL_ERROR, f"Goal verification failed: {error}")
            return
        if complete:
            self.mode = "complete"
            self.status = "Goal verified from observed state"
            self.events.append(
                "discovery_episode_complete",
                executed_actions=self._executed_actions,
                replans=self._replans,
            )
            return
        event = ReplanEvent(
            "goal_check",
            "goal_not_satisfied",
            "The current plan finished, but the observed goal remains unsatisfied.",
            self.completed_actions[-1] if self.completed_actions else None,
        )
        self.events.append("goal_replan_requested", **event.as_dict())
        self._begin_replan(event)

    def _begin_replan(self, event: ReplanEvent) -> None:
        if self.max_model_calls is not None and self._model_calls >= self.max_model_calls:
            self._fail(FailureCode.INFERENCE_FAILED, "Model-call budget exhausted")
            return
        if self._replans >= self.max_replans:
            self._fail(FailureCode.INFERENCE_FAILED, "Replan budget exhausted")
            return
        self._replans += 1
        if not self._observe("replan_observation"):
            return
        self._request_plan(event)

    def _request_plan(self, event: ReplanEvent | None) -> None:
        if self.snapshot is None:
            self._fail(FailureCode.INTERNAL_ERROR, "Cannot plan without an observation")
            return
        self.mode = "planning"
        self.status = "Requesting initial plan" if event is None else f"Replanning after {event.event_id}"
        request = PlannerRequest(
            self.scene,
            self.goal,
            self.snapshot,
            tuple(self.completed_actions),
            self.remaining_actions,
            event,
        )
        self._model_calls += 1
        try:
            result = self.planner.plan(request)
        except RecoverablePlanningError as error:
            event = ReplanEvent(
                "planning",
                "invalid_planner_output",
                str(error),
                None,
                {"planner_error": str(error)},
            )
            self.events.append("planning_replan_requested", **event.as_dict())
            self._begin_replan(event)
            return
        except Exception as error:
            self._fail(FailureCode.INFERENCE_FAILED, f"Planner request failed: {error}")
            return
        if result.latency_s is not None:
            self._planning_latency_s += float(result.latency_s)
        self.events.append(
            "planner_result",
            status=result.status.value,
            action_count=len(result.actions),
            latency_s=result.latency_s,
            message=result.message,
            replan_event=event.as_dict() if event else None,
        )
        self.last_event = event
        if result.status is PlanStatus.NO_VALID_PLAN:
            self._fail(FailureCode.INFERENCE_FAILED, result.message or "Planner returned no valid plan")
            return
        if result.status is PlanStatus.GOAL_COMPLETE:
            self._finish_plan()
            return
        if not result.actions:
            self._fail(FailureCode.INFERENCE_FAILED, "Planner returned PLAN with no actions")
            return
        prepare = getattr(self.dispatcher, "prepare", None)
        if callable(prepare):
            try:
                prepared = prepare(result.actions)
            except Exception as error:
                prepared = SkillResult.failed(FailureCode.INTERNAL_ERROR, str(error), recoverable=False)
            if prepared is not None and not prepared.success:
                record = _action_record(result.actions[0], prepared)
                self._record(result.actions[0], prepared)
                self._recover_or_fail(result.actions[0], record, recoverable=prepared.recoverable)
                return
        self._actions = result.actions
        self._action_index = 0
        self._active = False
        self.mode = "executing"
        self.status = f"Executing {len(self._actions)} planned skills"
        self.events.append(
            "plan_accepted",
            is_replan=event is not None,
            actions=[{"name": item.name, "arguments": dict(item.arguments)} for item in result.actions],
        )

    def _record(self, action: SkillAction, result: SkillResult) -> None:
        record = _action_record(action, result)
        self.history.append(record)
        self.events.append("discovery_skill_finished", **record)

    def _fail(self, code: FailureCode, message: str) -> None:
        self.mode = "failed"
        self.status = f"Discovery replanning failed: {message}"
        self.events.append("discovery_episode_failed", failure_code=code.value, message=message)
