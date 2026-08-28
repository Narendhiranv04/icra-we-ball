"""Failure-aware functional task execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from mujoco_scenes.foundation_model import (
    AssessmentBackend,
    Candidate,
    FunctionalAssessmentResult,
    RankingRequest,
)
from mujoco_scenes.tamp.candidates import visible_candidates
from mujoco_scenes.tamp.events import EventLog
from mujoco_scenes.tamp.functions import FunctionRegistry
from mujoco_scenes.tamp.skills import (
    FailureCode,
    SkillAction,
    SkillDispatcher,
    SkillResult,
    SkillStartError,
)
from mujoco_scenes.tamp.state import ObservedState


@dataclass(frozen=True)
class FunctionalTask:
    task_id: str
    subject_id: str
    required_function: str
    goal_relation: str


class PlanRejected(RuntimeError):
    def __init__(self, code: FailureCode, message: str):
        super().__init__(message)
        self.code = code


Observer = Callable[[], ObservedState]
PlanFactory = Callable[
    [FunctionalTask, Candidate, ObservedState], Sequence[SkillAction]
]
GoalVerifier = Callable[
    [FunctionalTask, Candidate, ObservedState], bool
]
DiscoveryPolicy = Callable[
    [FunctionalTask, ObservedState, int], Sequence[SkillAction]
]


class TaskExecutive:
    """Assess alternatives, execute one plan, and backtrack on failure."""

    TERMINAL_MODES = {"idle", "complete", "failed"}

    def __init__(
        self,
        registry: FunctionRegistry,
        assessment_backend: AssessmentBackend,
        observer: Observer,
        dispatcher: SkillDispatcher,
        plan_factory: PlanFactory,
        goal_verifier: GoalVerifier,
        *,
        discovery_policy: DiscoveryPolicy | None = None,
        max_discoveries: int = 0,
        event_log: EventLog | None = None,
    ):
        if (
            isinstance(max_discoveries, bool)
            or not isinstance(max_discoveries, int)
            or max_discoveries < 0
        ):
            raise ValueError("max_discoveries must be a non-negative integer")
        self.registry = registry
        self.assessment_backend = assessment_backend
        self.observer = observer
        self.dispatcher = dispatcher
        self.plan_factory = plan_factory
        self.goal_verifier = goal_verifier
        self.discovery_policy = discovery_policy
        self.max_discoveries = max_discoveries
        self.events = EventLog() if event_log is None else event_log
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tamp-inference"
        )
        self.mode = "idle"
        self.status = "TAMP executive idle"
        self.failure_code: FailureCode | None = None
        self.task: FunctionalTask | None = None
        self.state: ObservedState | None = None
        self.assessment: FunctionalAssessmentResult | None = None
        self._future: Future[FunctionalAssessmentResult] | None = None
        self._candidates: dict[str, Candidate] = {}
        self._candidate_queue: list[str] = []
        self._candidate: Candidate | None = None
        self._actions: tuple[SkillAction, ...] = ()
        self._action_index = 0
        self._action_active = False
        self._purpose = ""
        self._discoveries = 0

    @property
    def busy(self) -> bool:
        return self.mode not in self.TERMINAL_MODES

    @property
    def progress(self) -> float:
        if self.mode == "complete":
            return 1.0
        if self.mode == "assessing":
            return 0.1
        if not self._actions:
            return 0.0
        return min(0.95, self._action_index / len(self._actions))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def start(self, task: FunctionalTask) -> None:
        if self.busy:
            raise RuntimeError("A functional task is already running")
        self.task = task
        self.failure_code = None
        self.assessment = None
        self._candidate_queue.clear()
        self._candidate = None
        self._discoveries = 0
        self.events.append("task_started", task=task.task_id)
        self._begin_assessment()

    def _begin_assessment(self) -> None:
        if self.task is None:
            self._fail(FailureCode.INTERNAL_ERROR, "Functional task is missing")
            return
        try:
            self.state = self.observer()
            function = self.registry.get(self.task.required_function)
            candidates = visible_candidates(self.state, function)
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Candidate observation failed: {error}",
            )
            return
        self._candidates = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        self.events.append(
            "observation",
            revision=self.state.revision,
            visible_candidate_ids=list(self._candidates),
            state=self.state.as_dict(),
        )
        if not candidates:
            self._start_discovery_or_fail()
            return

        target = self.state.visible_object(self.task.subject_id)
        request = RankingRequest(
            self.task.required_function,
            candidates,
            target=target.as_dict() if target else None,
        )
        self._future = self._pool.submit(
            self.assessment_backend.assess, request
        )
        self.mode = "assessing"
        self.status = (
            f"Assessing {len(candidates)} visible "
            f"{self.task.required_function} candidates"
        )

    def _accept_assessment(self) -> None:
        if self._future is None:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                "Assessment completed without an active request",
            )
            return
        try:
            self.assessment = self._future.result()
        except Exception as error:
            self._fail(FailureCode.INFERENCE_FAILED, str(error))
            return
        self._future = None
        self._candidate_queue = list(
            self.assessment.ranked_candidate_ids
        )
        self.events.append(
            "functional_assessment",
            functional_candidate_ids=list(
                self.assessment.functional_candidate_ids
            ),
            ranked_candidate_ids=list(self._candidate_queue),
            model=self.assessment.model,
            latency_ms=self.assessment.latency_ms,
        )
        if not self._candidate_queue:
            self._start_discovery_or_fail()
            return
        self._start_next_candidate()

    def _start_next_candidate(self) -> None:
        if self.task is None or self.state is None:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                "Candidate selection has no active task observation",
            )
            return
        while self._candidate_queue:
            candidate_id = self._candidate_queue.pop(0)
            self._candidate = self._candidates.get(candidate_id)
            if self._candidate is None:
                self._fail(
                    FailureCode.INFERENCE_FAILED,
                    f"Assessment returned unknown candidate {candidate_id!r}",
                )
                return
            try:
                actions = tuple(
                    self.plan_factory(
                        self.task, self._candidate, self.state
                    )
                )
            except PlanRejected as error:
                self.events.append(
                    "candidate_rejected",
                    candidate=candidate_id,
                    failure_code=error.code.value,
                    message=str(error),
                )
                continue
            except Exception as error:
                self._fail(
                    FailureCode.INTERNAL_ERROR,
                    f"Candidate planning failed: {error}",
                )
                return
            if not actions:
                self.events.append(
                    "candidate_rejected",
                    candidate=candidate_id,
                    failure_code=FailureCode.PRECONDITION_FAILED.value,
                    message="planner returned no actions",
                )
                continue
            if self._start_actions(actions, "candidate"):
                self.events.append(
                    "candidate_selected",
                    candidate=candidate_id,
                    action_count=len(actions),
                )
            return
        self._start_discovery_or_fail()

    def _start_actions(
        self, actions: Sequence[SkillAction], purpose: str
    ) -> bool:
        self._actions = tuple(actions)
        prepare = getattr(self.dispatcher, "prepare", None)
        if callable(prepare):
            try:
                preparation = prepare(self._actions)
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
                    "plan_preparation_failed",
                    purpose=purpose,
                    failure_code=(
                        preparation.failure_code.value
                        if preparation.failure_code
                        else FailureCode.INTERNAL_ERROR.value
                    ),
                    message=preparation.message,
                )
                if purpose == "candidate":
                    self._candidate_failed(preparation)
                elif preparation.recoverable:
                    self._start_discovery_or_fail()
                else:
                    self._fail(
                        preparation.failure_code or FailureCode.INTERNAL_ERROR,
                        preparation.message,
                    )
                return False
        self._action_index = 0
        self._action_active = False
        self._purpose = purpose
        self.mode = "executing"
        self.status = f"Executing {purpose} plan"
        return True

    def _start_discovery_or_fail(self) -> None:
        if (
            self.discovery_policy is None
            or self._discoveries >= self.max_discoveries
        ):
            self._fail(
                FailureCode.NO_CANDIDATE,
                "No functional visible candidate remains",
            )
            return
        if self.task is None:
            self._fail(FailureCode.INTERNAL_ERROR, "Functional task is missing")
            return
        try:
            self.state = self.observer()
            actions = tuple(
                self.discovery_policy(
                    self.task, self.state, self._discoveries
                )
            )
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Discovery planning failed: {error}",
            )
            return
        self._discoveries += 1
        if not actions:
            self._fail(
                FailureCode.NO_CANDIDATE,
                "Discovery policy returned no action",
            )
            return
        self.events.append(
            "discovery_started",
            attempt=self._discoveries,
            actions=[action.name for action in actions],
        )
        self._start_actions(actions, "discovery")

    def _candidate_failed(self, result: SkillResult) -> None:
        candidate_id = (
            self._candidate.candidate_id
            if self._candidate is not None
            else None
        )
        self.events.append(
            "candidate_failed",
            candidate=candidate_id,
            failure_code=(
                result.failure_code.value
                if result.failure_code is not None
                else FailureCode.INTERNAL_ERROR.value
            ),
            message=result.message,
        )
        if not result.recoverable:
            self._fail(
                result.failure_code or FailureCode.INTERNAL_ERROR,
                result.message,
            )
            return
        try:
            self.state = self.observer()
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Failure observation failed: {error}",
            )
            return
        self._start_next_candidate()

    def _finish_actions(self) -> None:
        if self._purpose == "discovery":
            self.events.append("discovery_complete")
            self._begin_assessment()
            return

        if self.task is None or self._candidate is None:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                "Candidate execution lost its active task",
            )
            return
        try:
            self.state = self.observer()
            complete = self.goal_verifier(
                self.task, self._candidate, self.state
            )
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Goal verification failed: {error}",
            )
            return
        if complete:
            self.mode = "complete"
            self.status = (
                f"Task complete using {self._candidate.candidate_id}"
            )
            self.events.append(
                "task_complete",
                task=self.task.task_id,
                candidate=self._candidate.candidate_id,
            )
            return
        self._candidate_failed(
            SkillResult.failed(
                FailureCode.EFFECT_NOT_OBSERVED,
                "Planned effects were not observed",
            )
        )

    def _update_action(self) -> None:
        action = self._actions[self._action_index]
        if not self._action_active:
            try:
                self.dispatcher.start(action)
            except SkillStartError as error:
                result = SkillResult.failed(
                    error.code,
                    str(error),
                    recoverable=error.recoverable,
                )
                if self._purpose == "candidate":
                    self._candidate_failed(result)
                else:
                    self._start_discovery_or_fail()
                return
            except Exception as error:
                result = SkillResult.failed(
                    FailureCode.INTERNAL_ERROR,
                    str(error),
                    recoverable=False,
                )
                if self._purpose == "candidate":
                    self._candidate_failed(result)
                else:
                    self._fail(FailureCode.INTERNAL_ERROR, str(error))
                return
            self._action_active = True
            self.status = f"Executing {action.name}"
            self.events.append(
                "skill_started",
                skill=action.name,
                arguments=dict(action.arguments),
            )

        try:
            result = self.dispatcher.update()
        except Exception as error:
            result = SkillResult.failed(
                FailureCode.INTERNAL_ERROR,
                str(error),
                recoverable=False,
            )
        if result is None:
            return
        self._action_active = False
        self.events.append(
            "skill_finished",
            skill=action.name,
            success=result.success,
            failure_code=(
                result.failure_code.value
                if result.failure_code is not None
                else None
            ),
            message=result.message,
        )
        if not result.success:
            if self._purpose == "candidate":
                self._candidate_failed(result)
            elif result.recoverable:
                self._start_discovery_or_fail()
            else:
                self._fail(
                    result.failure_code or FailureCode.INTERNAL_ERROR,
                    result.message,
                )
            return
        try:
            self.state = self.observer()
        except Exception as error:
            self._fail(
                FailureCode.INTERNAL_ERROR,
                f"Post-skill observation failed: {error}",
            )
            return
        self.events.append(
            "observation",
            revision=self.state.revision,
            after_skill=action.name,
            state=self.state.as_dict(),
        )
        self._action_index += 1
        if self._action_index >= len(self._actions):
            self._finish_actions()

    def _fail(self, code: FailureCode, message: str) -> None:
        self.mode = "failed"
        self.failure_code = code
        self.status = f"Task failed: {message}"
        self.events.append(
            "task_failed", failure_code=code.value, message=message
        )

    def update(self) -> None:
        if self.mode == "assessing":
            if self._future is not None and self._future.done():
                self._accept_assessment()
            return
        if self.mode == "executing":
            self._update_action()
