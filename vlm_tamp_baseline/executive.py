"""Bounded VLM subgoal, TAMP refinement, and reprompting loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from baseline_common.inference import ModelTransportError, PlanningError
from baseline_common.models import Action, ActionResult, Observation

from .models import ObjectReference, ObjectUniverse, RefinementFailure, Subgoal
from .planner import SubgoalPlanResult
from .refiner import RefinementResult, subgoal_satisfied


class Planner(Protocol):
    def plan(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        *,
        universe: ObjectUniverse | None = None,
        succeeded_subgoals: Sequence[Subgoal] = (),
        action_history: Sequence[Mapping[str, Any]] = (),
        failure: RefinementFailure | None = None,
    ) -> SubgoalPlanResult: ...


class Refiner(Protocol):
    def refine(
        self, subgoal: Subgoal, observation: Observation
    ) -> RefinementResult: ...


class Executor(Protocol):
    def execute(self, action: Action) -> ActionResult: ...


@dataclass(frozen=True)
class ObservationFrame:
    observation: Observation
    images: tuple[Mapping[str, str], ...]


Observer = Callable[[], ObservationFrame]
StateObserver = Callable[[], Observation]
GoalVerifier = Callable[[Observation], bool]
SubgoalVerifier = Callable[[Subgoal, Observation], bool]
RefinementSink = Callable[[Subgoal, tuple[Action, ...]], None]


@dataclass(frozen=True)
class BaselineResult:
    success: bool
    status: str
    model_calls: int
    reprompts: int
    attempted_subgoals: int
    refined_subgoals: int
    executed_actions: int
    succeeded_subgoals: tuple[Subgoal, ...]
    action_history: tuple[Mapping[str, Any], ...]
    terminal_failure: RefinementFailure | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "model_calls": self.model_calls,
            "reprompts": self.reprompts,
            "attempted_subgoals": self.attempted_subgoals,
            "refined_subgoals": self.refined_subgoals,
            "executed_actions": self.executed_actions,
            "succeeded_subgoals": [item.as_dict() for item in self.succeeded_subgoals],
            "action_history": list(self.action_history),
            "terminal_failure": (
                self.terminal_failure.as_dict() if self.terminal_failure else None
            ),
        }


class VLMTAMPExecutive:
    """Reprompt only when a proposed subgoal cannot be refined or executed."""

    def __init__(
        self,
        planner: Planner,
        observer: Observer,
        executor: Executor,
        *,
        refiner: Refiner | None = None,
        goal_verifier: GoalVerifier | None = None,
        state_observer: StateObserver | None = None,
        subgoal_verifier: SubgoalVerifier = subgoal_satisfied,
        object_universe: ObjectUniverse | None = None,
        refinement_sink: RefinementSink | None = None,
        max_model_calls: int = 3,
        max_total_actions: int = 40,
        max_transport_retries: int = 2,
    ):
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in (max_model_calls, max_total_actions)
        ):
            raise ValueError("Executive limits must be positive")
        if (
            isinstance(max_transport_retries, bool)
            or not isinstance(max_transport_retries, int)
            or max_transport_retries < 0
        ):
            raise ValueError("max_transport_retries must be a non-negative integer")
        self.planner = planner
        self.observer = observer
        self.executor = executor
        if refiner is None:
            raise ValueError(
                "A refiner is required. Use PDDLStreamSubgoalRefiner for the "
                "paper baseline or CatalogSubgoalRefiner only for its named ablation."
            )
        self.refiner = refiner
        self.goal_verifier = goal_verifier or (lambda item: item.goal_satisfied)
        self.state_observer = state_observer or (
            lambda: self.observer().observation
        )
        self.subgoal_verifier = subgoal_verifier
        self.object_universe = object_universe
        self.refinement_sink = refinement_sink
        self.max_model_calls = max_model_calls
        self.max_total_actions = max_total_actions
        self.max_transport_retries = max_transport_retries

    def run(self, goal: str) -> BaselineResult:
        succeeded: list[Subgoal] = []
        history: list[Mapping[str, Any]] = []
        failure: RefinementFailure | None = None
        attempted = refined = executed = 0
        observed_objects: set[str] = set()
        model_call = 0
        transport_retries = 0

        while model_call < self.max_model_calls:
            frame = self.observer()
            if self.goal_verifier(frame.observation):
                return self._result(True, "GOAL_COMPLETE", model_call, attempted, refined, executed, succeeded, history)
            for item in frame.observation.entities:
                observed_objects.add(item.entity_id)
            universe = self.object_universe or ObjectUniverse(
                tuple(ObjectReference(key) for key in sorted(observed_objects))
            )
            try:
                proposal = self.planner.plan(
                    goal,
                    frame.observation,
                    frame.images,
                    universe=universe,
                    succeeded_subgoals=succeeded,
                    action_history=tuple(
                        {
                            key: value
                            for key, value in row.items()
                            if key != "details"
                        }
                        for row in history
                    ),
                    failure=failure,
                ).plan
            except ModelTransportError as error:
                # A transport failure produced no completion, so it is an
                # infrastructure fault rather than a model call.  It draws on
                # its own retry budget and never reports the episode as a
                # model-call budget exhaustion.
                failure = RefinementFailure(
                    "inference_failed",
                    str(error),
                )
                if transport_retries >= self.max_transport_retries:
                    return self._result(
                        False, "INFERENCE_FAILED", model_call, attempted,
                        refined, executed, succeeded, history, failure,
                    )
                transport_retries += 1
                continue
            except PlanningError as error:
                model_call += 1
                failure = RefinementFailure(
                    "invalid_vlm_output",
                    str(error),
                )
                continue
            model_call += 1
            if proposal.status == "NO_VALID_SUBGOALS":
                failure = RefinementFailure(
                    "no_valid_subgoals", "The VLM proposed no refinable subgoals."
                )
                continue
            if proposal.status == "GOAL_COMPLETE":
                refreshed = self.state_observer()
                if self.goal_verifier(refreshed):
                    return self._result(True, "GOAL_COMPLETE", model_call, attempted, refined, executed, succeeded, history)
                failure = RefinementFailure(
                    "effect_not_observed",
                    "The VLM declared completion but the goal verifier is false.",
                )
                continue

            failure = None
            replan = False
            for subgoal in proposal.subgoals:
                attempted += 1
                observation = self.state_observer()
                if self.subgoal_verifier(subgoal, observation):
                    if subgoal not in succeeded:
                        succeeded.append(subgoal)
                    continue
                refinement = self.refiner.refine(subgoal, observation)
                if not refinement.success:
                    failure = refinement.failure
                    replan = True
                    break
                refined += 1
                if self.refinement_sink is not None:
                    self.refinement_sink(subgoal, refinement.actions)
                prepare = getattr(self.executor, "prepare", None)
                if callable(prepare):
                    try:
                        preparation = prepare(refinement.actions)
                    except Exception as error:
                        preparation = ActionResult.failed(
                            "internal_error",
                            f"Plan preparation raised {type(error).__name__}: {error}",
                            recoverable=False,
                        )
                    if preparation is not None and not preparation.success:
                        failure = RefinementFailure(
                            preparation.failure_code or "execution_failed",
                            preparation.message or "Plan preparation failed.",
                            subgoal,
                        )
                        if not preparation.recoverable:
                            return self._result(False, "NON_RECOVERABLE_FAILURE", model_call, attempted, refined, executed, succeeded, history, failure)
                        replan = True
                        break
                for action in refinement.actions:
                    if executed >= self.max_total_actions:
                        failure = RefinementFailure(
                            "action_budget_exhausted",
                            f"The {self.max_total_actions}-action budget was exhausted.",
                            subgoal,
                        )
                        return self._result(False, "ACTION_BUDGET_EXHAUSTED", model_call, attempted, refined, executed, succeeded, history, failure)
                    result = self.executor.execute(action)
                    executed += 1
                    history.append(
                        {
                            "action": action.as_dict(),
                            "success": result.success,
                            "failure_code": result.failure_code,
                            "message": result.message,
                            "effects": list(result.effects),
                            "details": dict(result.details),
                        }
                    )
                    if not result.success:
                        failure = RefinementFailure(
                            result.failure_code or "execution_failed",
                            result.message or "Motion refinement or execution failed.",
                            subgoal,
                        )
                        if not result.recoverable:
                            return self._result(False, "NON_RECOVERABLE_FAILURE", model_call, attempted, refined, executed, succeeded, history, failure)
                        replan = True
                        break
                if replan:
                    break
                refreshed = self.state_observer()
                if not self.subgoal_verifier(subgoal, refreshed):
                    failure = RefinementFailure(
                        "effect_not_observed",
                        "The refined skill sequence finished but its subgoal effect was not observed.",
                        subgoal,
                    )
                    replan = True
                    break
                if subgoal not in succeeded:
                    succeeded.append(subgoal)
                if self.goal_verifier(refreshed):
                    return self._result(True, "GOAL_COMPLETE", model_call, attempted, refined, executed, succeeded, history)
            if replan:
                continue
            failure = RefinementFailure(
                "goal_not_satisfied",
                "All proposed subgoals succeeded but the task goal remains false.",
            )

        return self._result(False, "MODEL_CALL_BUDGET_EXHAUSTED", self.max_model_calls, attempted, refined, executed, succeeded, history, failure)

    def _result(
        self,
        success: bool,
        status: str,
        model_calls: int,
        attempted: int,
        refined: int,
        executed: int,
        succeeded: Sequence[Subgoal],
        history: Sequence[Mapping[str, Any]],
        failure: RefinementFailure | None = None,
    ) -> BaselineResult:
        return BaselineResult(
            success,
            status,
            model_calls,
            max(0, model_calls - 1),
            attempted,
            refined,
            executed,
            tuple(succeeded),
            tuple(history),
            failure,
        )
