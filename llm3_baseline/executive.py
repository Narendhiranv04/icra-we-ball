"""Bounded motion-failure replanning loop for the LLM3-style baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from baseline_common.inference import PlanningError

from .models import Action, ActionResult, Failure, Observation
from .planner import PlanResult


@dataclass(frozen=True)
class ObservationFrame:
    observation: Observation
    images: tuple[Mapping[str, str], ...]


class Planner(Protocol):
    def plan(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        *,
        history: Sequence[Mapping[str, Any]] = (),
        failure: Failure | None = None,
    ) -> PlanResult:
        """Propose the next bounded action sequence."""


class Executor(Protocol):
    def execute(self, action: Action) -> ActionResult:
        """Execute one validated action through the shared skill layer."""


Observer = Callable[[], ObservationFrame]
StateObserver = Callable[[], Observation]
GoalVerifier = Callable[[Observation], bool]


@dataclass(frozen=True)
class BaselineResult:
    success: bool
    status: str
    model_calls: int
    executed_actions: int
    history: tuple[Mapping[str, Any], ...]
    planning_trace: tuple[Mapping[str, Any], ...] = ()
    terminal_failure: Failure | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "model_calls": self.model_calls,
            "executed_actions": self.executed_actions,
            "history": list(self.history),
            "planning_trace": list(self.planning_trace),
            "terminal_failure": (
                self.terminal_failure.as_dict() if self.terminal_failure else None
            ),
        }


class LLM3Executive:
    """Execute VLM plans and return observed failures for re-planning."""

    def __init__(
        self,
        planner: Planner,
        observer: Observer,
        executor: Executor,
        *,
        goal_verifier: GoalVerifier | None = None,
        state_observer: StateObserver | None = None,
        max_model_calls: int = 5,
        max_total_actions: int = 30,
        trace_size: int = 3,
    ):
        limits = (max_model_calls, max_total_actions, trace_size)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in limits
        ):
            raise ValueError("Executive limits must be positive")
        self.planner = planner
        self.observer = observer
        self.executor = executor
        self.goal_verifier = goal_verifier or (
            lambda observation: observation.goal_satisfied
        )
        self.state_observer = state_observer or (
            lambda: self.observer().observation
        )
        self.max_model_calls = max_model_calls
        self.max_total_actions = max_total_actions
        self.trace_size = trace_size

    def run(self, goal: str) -> BaselineResult:
        history: list[Mapping[str, Any]] = []
        planning_trace: list[Mapping[str, Any]] = []
        failure: Failure | None = None
        executed = 0

        for model_call in range(1, self.max_model_calls + 1):
            frame = self.observer()
            if self.goal_verifier(frame.observation):
                return self._result(
                    True, "GOAL_COMPLETE", model_call - 1, executed, history,
                    planning_trace=planning_trace,
                )
            try:
                proposed = self.planner.plan(
                    goal,
                    frame.observation,
                    frame.images,
                    history=tuple(planning_trace[-self.trace_size :]),
                    failure=failure,
                ).plan
            except PlanningError as error:
                failure = Failure("invalid_model_output", str(error))
                planning_trace.append(
                    {
                        "full_plan": None,
                        "motion_feedback": [],
                        "planning_failure": failure.as_dict(),
                    }
                )
                continue
            planning_trace.append(
                {
                    "full_plan": proposed.as_dict(),
                    "motion_feedback": [],
                }
            )
            if proposed.status == "NO_VALID_PLAN":
                terminal = Failure(
                    "no_valid_plan",
                    "The planner reported no valid observed-state plan.",
                )
                return self._result(
                    False,
                    "NO_VALID_PLAN",
                    model_call,
                    executed,
                    history,
                    planning_trace=planning_trace,
                    failure=terminal,
                )
            precondition_failure = self._plan_precondition_failure(
                proposed.actions, frame.observation
            )
            if precondition_failure is not None:
                failure = precondition_failure
                continue

            prepare = getattr(self.executor, "prepare", None)
            if callable(prepare):
                try:
                    preparation = prepare(proposed.actions)
                except Exception as error:
                    preparation = ActionResult.failed(
                        "internal_error",
                        f"Plan preparation raised {type(error).__name__}: {error}",
                        recoverable=False,
                    )
                if preparation is not None and not preparation.success:
                    failure = Failure(
                        preparation.failure_code or "execution_failed",
                        preparation.message or "Plan preparation failed.",
                    )
                    planning_trace[-1]["motion_feedback"].append(
                        {
                            "stage": "plan_preparation",
                            "success": False,
                            "failure_code": preparation.failure_code,
                            "message": preparation.message,
                        }
                    )
                    if not preparation.recoverable:
                        return self._result(
                            False,
                            "NON_RECOVERABLE_FAILURE",
                            model_call,
                            executed,
                            history,
                            planning_trace=planning_trace,
                            failure=failure,
                        )
                    continue

            failure = None
            force_replan = False
            for action_index, action in enumerate(proposed.actions):
                if executed >= self.max_total_actions:
                    terminal = Failure(
                        "action_budget_exhausted",
                        f"The {self.max_total_actions}-action budget was exhausted.",
                        action,
                        action_index,
                    )
                    return self._result(
                        False,
                        "ACTION_BUDGET_EXHAUSTED",
                        model_call,
                        executed,
                        history,
                        planning_trace=planning_trace,
                        failure=terminal,
                    )
                result = self.executor.execute(action)
                executed += 1
                record = {
                    "action": action.as_dict(),
                    "success": result.success,
                    "failure_code": result.failure_code,
                    "message": result.message,
                    "effects": list(result.effects),
                    "details": dict(result.details),
                }
                history.append(record)
                planning_trace[-1]["motion_feedback"].append(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "details"
                    }
                )
                if not result.success:
                    failure = Failure(
                        result.failure_code or "execution_failed",
                        result.message or "Action execution failed.",
                        action,
                        action_index,
                    )
                    if not result.recoverable:
                        return self._result(
                            False,
                            "NON_RECOVERABLE_FAILURE",
                            model_call,
                            executed,
                            history,
                            planning_trace=planning_trace,
                            failure=failure,
                        )
                    force_replan = True
                    break

                refreshed = self.state_observer()
                if self.goal_verifier(refreshed):
                    return self._result(
                        True,
                        "GOAL_COMPLETE",
                        model_call,
                        executed,
                        history,
                        planning_trace=planning_trace,
                    )
                if action.skill == "INSPECT":
                    force_replan = True
                    break

            if force_replan:
                continue
            failure = Failure(
                "goal_not_satisfied",
                "The proposed action sequence finished but the goal remains false.",
            )

        terminal = failure or Failure(
            "model_call_budget_exhausted",
            f"The {self.max_model_calls}-call model budget was exhausted.",
        )
        return self._result(
            False,
            "MODEL_CALL_BUDGET_EXHAUSTED",
            self.max_model_calls,
            executed,
            history,
            planning_trace=planning_trace,
            failure=terminal,
        )

    @staticmethod
    def _plan_precondition_failure(
        actions: Sequence[Action], observation: Observation
    ) -> Failure | None:
        """Validate hand-state transitions before authorizing any motion."""
        held = observation.robot.get("held_object")
        if held is None:
            held = observation.robot.get("holding")
        held = str(held) if isinstance(held, str) and held else None

        for index, action in enumerate(actions):
            skill = action.skill.upper()
            values = action.arguments
            required_held = None
            if skill == "PICK":
                if held is not None:
                    return Failure(
                        "precondition_failed",
                        f"PICK requires an empty gripper, but {held} is held. "
                        f"PLACE {held} before PICK.",
                        action,
                        index,
                    )
                held = values["object_id"]
                continue
            if skill == "PLACE":
                required_held = values["object_id"]
                if values["object_id"] == values["region_id"]:
                    return Failure(
                        "precondition_failed",
                        "PLACE cannot place an object into itself.",
                        action,
                        index,
                    )
            elif skill == "POUR":
                required_held = values["source_id"]
                if values["source_id"] == values["target_id"]:
                    return Failure(
                        "precondition_failed",
                        "POUR requires distinct source and target objects.",
                        action,
                        index,
                    )
            elif skill in {"STIR", "CLEAN", "FASTEN"}:
                required_held = values["tool_id"]
                if skill == "STIR" and values["tool_id"] == values["target_id"]:
                    return Failure(
                        "precondition_failed",
                        "STIR requires distinct tool and target objects.",
                        action,
                        index,
                    )
            elif skill == "INSERT":
                required_held = values["fastener_id"]
            elif skill == "INSPECT" and held is not None:
                return Failure(
                    "precondition_failed",
                    f"INSPECT requires an empty gripper, but {held} is held. "
                    f"PLACE {held} before INSPECT.",
                    action,
                    index,
                )

            if required_held is not None and held != required_held:
                state = "empty" if held is None else f"holding {held}"
                return Failure(
                    "precondition_failed",
                    f"{skill} requires holding {required_held}, but the "
                    f"gripper is {state}. PICK {required_held} first",
                    action,
                    index,
                )
            if skill == "PLACE":
                held = None
        return None

    @staticmethod
    def _result(
        success: bool,
        status: str,
        model_calls: int,
        executed_actions: int,
        history: Sequence[Mapping[str, Any]],
        failure: Failure | None = None,
        *,
        planning_trace: Sequence[Mapping[str, Any]] = (),
    ) -> BaselineResult:
        return BaselineResult(
            success,
            status,
            model_calls,
            executed_actions,
            tuple(history),
            tuple(planning_trace),
            failure,
        )
