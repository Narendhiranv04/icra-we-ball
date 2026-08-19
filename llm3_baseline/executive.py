"""Bounded motion-failure replanning loop for the LLM3-style baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

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
GoalVerifier = Callable[[Observation], bool]


@dataclass(frozen=True)
class BaselineResult:
    success: bool
    status: str
    model_calls: int
    executed_actions: int
    history: tuple[Mapping[str, Any], ...]
    terminal_failure: Failure | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "model_calls": self.model_calls,
            "executed_actions": self.executed_actions,
            "history": list(self.history),
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
        max_model_calls: int = 5,
        max_total_actions: int = 30,
    ):
        if max_model_calls <= 0 or max_total_actions <= 0:
            raise ValueError("Executive limits must be positive")
        self.planner = planner
        self.observer = observer
        self.executor = executor
        self.goal_verifier = goal_verifier or (
            lambda observation: observation.goal_satisfied
        )
        self.max_model_calls = max_model_calls
        self.max_total_actions = max_total_actions

    def run(self, goal: str) -> BaselineResult:
        history: list[Mapping[str, Any]] = []
        failure: Failure | None = None
        executed = 0

        for model_call in range(1, self.max_model_calls + 1):
            frame = self.observer()
            if self.goal_verifier(frame.observation):
                return self._result(
                    True, "GOAL_COMPLETE", model_call - 1, executed, history
                )
            proposed = self.planner.plan(
                goal,
                frame.observation,
                frame.images,
                history=history,
                failure=failure,
            ).plan
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
                    terminal,
                )
            if proposed.status == "GOAL_COMPLETE":
                refreshed = self.observer().observation
                if self.goal_verifier(refreshed):
                    return self._result(
                        True, "GOAL_COMPLETE", model_call, executed, history
                    )
                failure = Failure(
                    "effect_not_observed",
                    "The model declared completion but the goal verifier is false.",
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
                        terminal,
                    )
                result = self.executor.execute(action)
                executed += 1
                record = {
                    "action": action.as_dict(),
                    "success": result.success,
                    "failure_code": result.failure_code,
                    "message": result.message,
                }
                history.append(record)
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
                            failure,
                        )
                    force_replan = True
                    break

                refreshed = self.observer().observation
                if self.goal_verifier(refreshed):
                    return self._result(
                        True, "GOAL_COMPLETE", model_call, executed, history
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
            terminal,
        )

    @staticmethod
    def _result(
        success: bool,
        status: str,
        model_calls: int,
        executed_actions: int,
        history: Sequence[Mapping[str, Any]],
        failure: Failure | None = None,
    ) -> BaselineResult:
        return BaselineResult(
            success,
            status,
            model_calls,
            executed_actions,
            tuple(history),
            failure,
        )
