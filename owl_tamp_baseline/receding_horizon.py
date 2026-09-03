"""Paper-described OWL-TAMP receding-horizon control.

The OWL-TAMP simulation experiments are single-shot.  Its real-robot
deployment repeatedly observes, plans, and executes from the current state.
This module implements that policy without inventing a failure-feedback prompt:
each new decision receives only the fresh observable state and the original
goal.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from baseline_common.models import ActionResult, Observation

from .models import Action, Constraint, PlanningResult


class Planner(Protocol):
    response_trace: list[dict[str, Any]]
    trace: Mapping[str, Any]

    def plan(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        oracle: Callable[[Action, Sequence[Constraint], int], bool],
        *,
        movable_object_ids: Sequence[str] | None = None,
        max_vlm_requests: int | None = None,
    ) -> PlanningResult: ...


Observe = Callable[[], tuple[Observation, Sequence[Mapping[str, str]]]]
Execute = Callable[[Action], ActionResult]
GoalVerifier = Callable[[Observation], bool]
Oracle = Callable[[Action, Sequence[Constraint], int], bool]
MovableObjects = Callable[[Observation], Sequence[str] | None]


@dataclass(frozen=True)
class RecedingHorizonResult:
    success: bool
    status: str
    planning_rounds: int
    replans: int
    raw_vlm_requests: int
    executed_actions: int
    action_history: tuple[Mapping[str, Any], ...]
    planning_trace: tuple[Mapping[str, Any], ...]
    failure: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "planning_rounds": self.planning_rounds,
            "replans": self.replans,
            "raw_vlm_requests": self.raw_vlm_requests,
            "executed_actions": self.executed_actions,
            "action_history": [dict(row) for row in self.action_history],
            "planning_trace": [dict(row) for row in self.planning_trace],
            "failure": self.failure,
        }


class OWLTAMPRecedingHorizon:
    """Replan from a fresh observation after every successful action.

    A replan is a new complete OWL-TAMP planning cycle after the first cycle.
    The initial sketch and its per-action constraint requests are deliberately
    not counted as replans.
    """

    def __init__(
        self,
        planner: Planner,
        observe: Observe,
        execute: Execute,
        goal_verifier: GoalVerifier,
        oracle: Oracle,
        *,
        movable_objects: MovableObjects | None = None,
        max_replans: int = 8,
        max_total_actions: int = 48,
    ):
        if max_replans < 0 or max_total_actions < 1:
            raise ValueError("max_replans must be non-negative and max_total_actions positive")
        self.planner = planner
        self.observe = observe
        self.execute = execute
        self.goal_verifier = goal_verifier
        self.oracle = oracle
        self.movable_objects = movable_objects or (lambda _observation: None)
        self.max_replans = max_replans
        self.max_total_actions = max_total_actions

    def run(self, goal: str) -> RecedingHorizonResult:
        history: list[Mapping[str, Any]] = []
        traces: list[Mapping[str, Any]] = []
        raw_requests = 0

        for planning_round in range(1, self.max_replans + 2):
            observation, images = self.observe()
            if self.goal_verifier(observation):
                return self._result(
                    True, "GOAL_COMPLETE", planning_round - 1, raw_requests, history, traces
                )

            result = self.planner.plan(
                goal,
                observation,
                images,
                self.oracle,
                movable_object_ids=self.movable_objects(observation),
                max_vlm_requests=None,
            )
            raw_requests += len(self.planner.response_trace)
            traces.append(
                {
                    "planning_round": planning_round,
                    "observation_revision": observation.revision,
                    "result": result.as_dict(),
                    "model_trace": deepcopy(dict(self.planner.trace)),
                }
            )
            if result.status != "PLAN" or not result.actions:
                return self._result(
                    False,
                    "NO_PLAN",
                    planning_round,
                    raw_requests,
                    history,
                    traces,
                    result.failure or "OWL-TAMP returned no executable plan.",
                )
            if len(history) >= self.max_total_actions:
                return self._result(
                    False,
                    "ACTION_BUDGET_EXHAUSTED",
                    planning_round,
                    raw_requests,
                    history,
                    traces,
                    f"The {self.max_total_actions}-action budget was exhausted.",
                )

            action = result.actions[0]
            outcome = self.execute(action)
            history.append(
                {
                    "planning_round": planning_round,
                    "action": action.as_dict(),
                    "success": outcome.success,
                    "failure_code": outcome.failure_code,
                    "message": outcome.message,
                    "effects": list(outcome.effects),
                    "details": dict(outcome.details),
                }
            )
            if not outcome.success:
                return self._result(
                    False,
                    "EXECUTION_FAILED",
                    planning_round,
                    raw_requests,
                    history,
                    traces,
                    outcome.message or outcome.failure_code or "Action execution failed.",
                )

        return self._result(
            False,
            "REPLAN_BUDGET_EXHAUSTED",
            self.max_replans + 1,
            raw_requests,
            history,
            traces,
            f"The {self.max_replans}-replan budget was exhausted.",
        )

    @staticmethod
    def _result(
        success: bool,
        status: str,
        planning_rounds: int,
        raw_vlm_requests: int,
        history: Sequence[Mapping[str, Any]],
        traces: Sequence[Mapping[str, Any]],
        failure: str = "",
    ) -> RecedingHorizonResult:
        return RecedingHorizonResult(
            success=success,
            status=status,
            planning_rounds=planning_rounds,
            replans=max(0, planning_rounds - 1),
            raw_vlm_requests=raw_vlm_requests,
            executed_actions=len(history),
            action_history=tuple(history),
            planning_trace=tuple(traces),
            failure=failure,
        )
