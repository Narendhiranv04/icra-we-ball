"""Execution of a verified functional witness through a deterministic sequencer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .events import EventLog
from .skills import FailureCode, SkillAction, SkillDispatcher, SkillResult, SkillStartError
from .state import ObservedState


@dataclass(frozen=True)
class GroundedTask:
    """Goal plus the semantic/geometric witness selected by search."""

    task_id: str
    goal: str
    witness: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionFailure:
    code: FailureCode
    message: str
    action: SkillAction | None = None


class Sequencer(Protocol):
    def sequence(
        self,
        task: GroundedTask,
        state: ObservedState,
        *,
        history: Sequence[Mapping[str, Any]],
        failure: ExecutionFailure | None,
    ) -> Sequence[SkillAction]:
        """Return grounded actions using only the verified handoff and state."""


Observer = Callable[[], ObservedState]
GoalVerifier = Callable[
    [GroundedTask, ObservedState, Sequence[Mapping[str, Any]]], bool
]


class GroundedPlanExecutive:
    """Run one verified handoff without invoking a foundation model.

    Search and FM decomposition happen before this boundary. Recoverable
    failures are returned only to the deterministic sequencer.
    """

    TERMINAL_MODES = {"idle", "complete", "failed"}

    def __init__(
        self,
        observer: Observer,
        sequencer: Sequencer,
        dispatcher: SkillDispatcher,
        goal_verifier: GoalVerifier,
        *,
        max_replans: int = 3,
        max_actions: int = 60,
        event_log: EventLog | None = None,
    ):
        if (
            isinstance(max_replans, bool)
            or not isinstance(max_replans, int)
            or isinstance(max_actions, bool)
            or not isinstance(max_actions, int)
            or max_replans < 0
            or max_actions <= 0
        ):
            raise ValueError("execution limits are invalid")
        self.observer = observer
        self.sequencer = sequencer
        self.dispatcher = dispatcher
        self.goal_verifier = goal_verifier
        self.max_replans = max_replans
        self.max_actions = max_actions
        self.events = EventLog() if event_log is None else event_log
        self.mode = "idle"
        self.status = "Grounded execution idle"
        self.task: GroundedTask | None = None
        self.state: ObservedState | None = None
        self.failure: ExecutionFailure | None = None
        self.history: list[Mapping[str, Any]] = []
        self._actions: tuple[SkillAction, ...] = ()
        self._action_index = 0
        self._active = False
        self._replans = 0
        self._executed = 0

    @property
    def busy(self) -> bool:
        return self.mode not in self.TERMINAL_MODES

    @property
    def replans(self) -> int:
        return self._replans

    @property
    def executed_actions(self) -> int:
        return self._executed

    def start(self, task: GroundedTask) -> None:
        if self.busy:
            raise RuntimeError("Grounded execution is already running")
        self.task = task
        self.failure = None
        self.history.clear()
        self._replans = 0
        self._executed = 0
        try:
            self.state = self.observer()
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Initial observation failed: {error}",
            )
            return
        self.events.append(
            "grounded_task_started",
            task=task.task_id,
            observed_revision=self.state.revision,
        )
        try:
            already_complete = self.goal_verifier(task, self.state, self.history)
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Goal verification failed: {error}",
            )
            return
        if already_complete:
            self.mode = "complete"
            self.status = "Goal already satisfied"
            return
        self._request_sequence(None)

    def _request_sequence(self, failure: ExecutionFailure | None) -> None:
        if self.task is None or self.state is None:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                "Grounded execution has no active task observation",
            )
            return
        try:
            actions = tuple(
                self.sequencer.sequence(
                    self.task,
                    self.state,
                    history=self.history,
                    failure=failure,
                )
            )
        except Exception as error:
            self._fail(FailureCode.INTERNAL_ERROR, str(error))
            return
        if not actions:
            self._fail(
                failure.code if failure else FailureCode.PRECONDITION_FAILED,
                "Deterministic sequencer returned no executable actions",
            )
            return
        prepare = getattr(self.dispatcher, "prepare", None)
        if callable(prepare):
            try:
                preparation = prepare(actions)
            except SkillStartError as error:
                preparation = SkillResult.failed(
                    error.code, str(error), recoverable=error.recoverable
                )
            except Exception as error:
                preparation = SkillResult.failed(
                    FailureCode.INTERNAL_ERROR, str(error), recoverable=False
                )
            if preparation is not None and not preparation.success:
                self.events.append(
                    "sequence_preparation_failed",
                    failure_code=(
                        preparation.failure_code.value
                        if preparation.failure_code
                        else FailureCode.INTERNAL_ERROR.value
                    ),
                    message=preparation.message,
                )
                self._handle_failure(actions[0], preparation)
                return
        self._actions = actions
        self._action_index = 0
        self._active = False
        self.mode = "executing"
        self.status = f"Executing {len(actions)} grounded actions"
        self.events.append(
            "sequence_selected",
            replan_index=self._replans,
            actions=[
                {"name": action.name, "arguments": dict(action.arguments)}
                for action in actions
            ],
        )

    def _record(self, action: SkillAction, result: SkillResult) -> None:
        record = {
            "action": {"name": action.name, "arguments": dict(action.arguments)},
            "success": result.success,
            "effects": list(result.effects),
            "failure_code": (
                result.failure_code.value if result.failure_code else None
            ),
            "message": result.message,
        }
        self.history.append(record)
        self.events.append("grounded_skill_finished", **record)

    def _handle_failure(self, action: SkillAction, result: SkillResult) -> None:
        failure = ExecutionFailure(
            result.failure_code or FailureCode.INTERNAL_ERROR,
            result.message or "Physical skill failed",
            action,
        )
        self.failure = failure
        if not result.recoverable:
            self._fail(failure.code, failure.message)
            return
        if self._replans >= self.max_replans:
            self._fail(failure.code, "Deterministic replan budget exhausted")
            return
        self._replans += 1
        try:
            self.state = self.observer()
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Replan observation failed: {error}",
            )
            return
        self.events.append(
            "deterministic_replan_requested",
            replan_index=self._replans,
            failure_code=failure.code.value,
        )
        self._request_sequence(failure)

    def _finish_sequence(self) -> None:
        if self.task is None:
            self._fail(FailureCode.INTERNAL_ERROR, "Grounded task is missing")
            return
        try:
            self.state = self.observer()
            complete = self.goal_verifier(self.task, self.state, self.history)
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Final goal verification failed: {error}",
            )
            return
        if complete:
            self.mode = "complete"
            self.status = "Grounded goal verified"
            self.failure = None
            self.events.append(
                "grounded_task_complete",
                task=self.task.task_id,
                executed_actions=self._executed,
                replans=self._replans,
            )
            return
        result = SkillResult.failed(
            FailureCode.EFFECT_NOT_OBSERVED,
            "The sequence completed but the observed goal is false",
        )
        self._handle_failure(self._actions[-1], result)

    def _fail(self, code: FailureCode, message: str) -> None:
        self.mode = "failed"
        self.status = f"Grounded execution failed: {message}"
        if self.failure is None:
            self.failure = ExecutionFailure(code, message)
        self.events.append(
            "grounded_task_failed", failure_code=code.value, message=message
        )

    def update(self) -> None:
        if self.mode != "executing":
            return
        if self._executed >= self.max_actions:
            self._fail(
                FailureCode.PRECONDITION_FAILED,
                "Grounded action budget exhausted",
            )
            return
        action = self._actions[self._action_index]
        if not self._active:
            self.status = (
                f"Executing {action.name} "
                f"({self._action_index + 1}/{len(self._actions)})"
            )
            try:
                self.dispatcher.start(action)
            except SkillStartError as error:
                result = SkillResult.failed(
                    error.code, str(error), recoverable=error.recoverable
                )
                self._record(action, result)
                self._handle_failure(action, result)
                return
            except Exception as error:
                result = SkillResult.failed(
                    FailureCode.INTERNAL_ERROR,
                    str(error),
                    recoverable=False,
                )
                self._record(action, result)
                self._handle_failure(action, result)
                return
            self._active = True
            self.events.append(
                "grounded_skill_started",
                action=action.name,
                arguments=dict(action.arguments),
            )
        try:
            result = self.dispatcher.update()
        except Exception as error:
            result = SkillResult.failed(
                FailureCode.INTERNAL_ERROR, str(error), recoverable=False
            )
        if result is None:
            return
        self._active = False
        self._executed += 1
        self._record(action, result)
        try:
            self.state = self.observer()
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Post-action observation failed: {error}",
            )
            return
        if not result.success:
            self._handle_failure(action, result)
            return
        self._action_index += 1
        if self._action_index >= len(self._actions):
            self._finish_sequence()


class FixedSequence:
    """Adapter for an already-generated deterministic sequence artifact."""

    def __init__(self, actions: Sequence[SkillAction]):
        self.actions = tuple(actions)

    def sequence(
        self,
        _task: GroundedTask,
        _state: ObservedState,
        *,
        history: Sequence[Mapping[str, Any]],
        failure: ExecutionFailure | None,
    ) -> Sequence[SkillAction]:
        if failure is not None:
            return ()
        return self.actions
