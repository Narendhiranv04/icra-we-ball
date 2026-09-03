from __future__ import annotations

from dataclasses import dataclass

from mujoco_scenes.tamp.discovery_replanning import (
    DiscoveryReplanningExecutive,
    PlanStatus,
    PlannerRequest,
    PlannerResult,
    RecoverablePlanningError,
)
from mujoco_scenes.tamp.skills import FailureCode, SkillAction, SkillResult
from mujoco_scenes.tamp.state import ObjectObservation, ObservedState, RobotObservation


def _state(revision: int, visible: tuple[str, ...], held: str | None = None) -> ObservedState:
    return ObservedState(
        {
            object_id: ObjectObservation(object_id, object_id, True, "table")
            for object_id in visible
        },
        {},
        RobotObservation("home", held),
        revision=revision,
    )


class ScriptedPlanner:
    def __init__(self, results: list[PlannerResult]):
        self.results = list(results)
        self.requests: list[PlannerRequest] = []

    def plan(self, request: PlannerRequest) -> PlannerResult:
        self.requests.append(request)
        return self.results.pop(0)


@dataclass
class FakeDispatcher:
    results: list[SkillResult]

    def __post_init__(self):
        self.started: list[SkillAction] = []
        self.active = False

    def prepare(self, _actions):
        return SkillResult.succeeded()

    def start(self, action: SkillAction) -> None:
        self.started.append(action)
        self.active = True

    def update(self):
        if not self.active:
            return None
        self.active = False
        return self.results.pop(0)


def _run(executive: DiscoveryReplanningExecutive) -> None:
    executive.start()
    for _ in range(50):
        if not executive.busy:
            return
        executive.update()
    raise AssertionError("executive did not terminate")


def test_new_visibility_interrupts_and_replans_from_fresh_state():
    states = iter(
        (
            _state(1, ("mug",)),
            _state(2, ("mug", "spoon")),
            _state(3, ("mug", "spoon")),
            _state(4, ("mug", "spoon")),
        )
    )
    planner = ScriptedPlanner(
        [
            PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "mug"}),)),
            PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "spoon"}),)),
        ]
    )
    dispatcher = FakeDispatcher([SkillResult.succeeded("picked(mug)"), SkillResult.succeeded("picked(spoon)")])
    executive = DiscoveryReplanningExecutive(
        scene="kitchen",
        goal="test",
        observer=lambda: next(states),
        planner=planner,
        dispatcher=dispatcher,
        goal_verifier=lambda _goal, state, _history: state.revision >= 4,
    )

    _run(executive)

    assert executive.mode == "complete"
    assert executive.replans == 1
    assert planner.requests[1].replan_event is not None
    assert planner.requests[1].replan_event.event_id == "new_object_discovered"
    assert planner.requests[1].replan_event.evidence["newly_visible_object_ids"] == ["spoon"]


def test_single_model_call_condition_executes_initial_plan_without_replanning():
    states = iter(
        (
            _state(1, ("mug",)),
            _state(2, ("mug", "spoon")),
        )
    )
    planner = ScriptedPlanner(
        [PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "mug"}),))]
    )
    executive = DiscoveryReplanningExecutive(
        scene="kitchen",
        goal="test",
        observer=lambda: next(states),
        planner=planner,
        dispatcher=FakeDispatcher([SkillResult.succeeded("picked(mug)")]),
        goal_verifier=lambda *_args: False,
        max_model_calls=1,
    )

    _run(executive)

    assert executive.mode == "failed"
    assert executive.model_calls == 1
    assert executive.planning_latency_s == 0.0
    assert executive.replans == 0
    assert len(planner.requests) == 1
    assert "Model-call budget exhausted" in executive.status


def test_recoverable_skill_failure_replans_with_failure_event():
    states = iter(
        (
            _state(1, ("mug",)),
            _state(2, ("mug",)),
            _state(3, ("mug",)),
            _state(4, ("mug",)),
        )
    )
    planner = ScriptedPlanner(
        [
            PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "mug"}),)),
            PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "mug"}),)),
        ]
    )
    dispatcher = FakeDispatcher(
        [
            SkillResult.failed(FailureCode.GRASP_FAILED, "grasp slipped"),
            SkillResult.succeeded("picked(mug)"),
        ]
    )
    executive = DiscoveryReplanningExecutive(
        scene="kitchen",
        goal="test",
        observer=lambda: next(states),
        planner=planner,
        dispatcher=dispatcher,
        goal_verifier=lambda _goal, state, _history: state.revision >= 4,
    )

    _run(executive)

    assert executive.mode == "complete"
    assert executive.replans == 1
    assert planner.requests[1].replan_event is not None
    assert planner.requests[1].replan_event.event_type == "failure"
    assert planner.requests[1].replan_event.event_id == FailureCode.GRASP_FAILED.value


def test_discovery_is_deferred_until_the_held_object_is_placed():
    states = iter(
        (
            _state(1, ("mug",)),
            _state(2, ("mug", "spoon"), held="mug"),
            _state(3, ("mug", "spoon")),
            _state(4, ("mug", "spoon")),
            _state(5, ("mug", "spoon")),
        )
    )
    planner = ScriptedPlanner(
        [
            PlannerResult(
                PlanStatus.PLAN,
                (
                    SkillAction("PICK", {"object_id": "mug"}),
                    SkillAction("PLACE", {"object_id": "mug", "region_id": "table"}),
                ),
            ),
            PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "spoon"}),)),
        ]
    )
    dispatcher = FakeDispatcher(
        [
            SkillResult.succeeded("picked(mug)"),
            SkillResult.succeeded("placed(mug,table)"),
            SkillResult.succeeded("picked(spoon)"),
        ]
    )
    executive = DiscoveryReplanningExecutive(
        scene="kitchen",
        goal="test",
        observer=lambda: next(states),
        planner=planner,
        dispatcher=dispatcher,
        goal_verifier=lambda _goal, state, _history: state.revision >= 5,
    )

    _run(executive)

    assert executive.mode == "complete"
    assert [item.name for item in dispatcher.started] == ["PICK", "PLACE", "PICK"]
    assert executive.replans == 1


def test_model_completion_is_not_accepted_without_observed_goal_completion():
    states = iter(
        (
            _state(1, ("mug",)),
            _state(2, ("mug",)),
            _state(3, ("mug",)),
        )
    )
    planner = ScriptedPlanner(
        [
            PlannerResult(PlanStatus.GOAL_COMPLETE),
            PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "mug"}),)),
        ]
    )
    dispatcher = FakeDispatcher([SkillResult.succeeded("picked(mug)")])
    executive = DiscoveryReplanningExecutive(
        scene="kitchen",
        goal="test",
        observer=lambda: next(states),
        planner=planner,
        dispatcher=dispatcher,
        goal_verifier=lambda _goal, state, _history: state.revision >= 3,
    )

    _run(executive)

    assert executive.mode == "complete"
    assert executive.replans == 1
    assert planner.requests[1].replan_event is not None
    assert planner.requests[1].replan_event.event_id == "goal_not_satisfied"


def test_effect_recording_error_becomes_a_typed_terminal_failure():
    states = iter((_state(1, ("mug",)), _state(2, ("mug",))))
    executive = DiscoveryReplanningExecutive(
        scene="kitchen",
        goal="test",
        observer=lambda: next(states),
        planner=ScriptedPlanner(
            [PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "mug"}),))]
        ),
        dispatcher=FakeDispatcher([SkillResult.succeeded("picked(mug)")]),
        goal_verifier=lambda *_args: False,
        effect_sink=lambda _effects: (_ for _ in ()).throw(RuntimeError("ledger unavailable")),
    )

    _run(executive)

    assert executive.mode == "failed"
    assert "Effect recording failed" in executive.status


def test_recoverable_planner_error_retries_from_a_fresh_observation():
    class RetryPlanner:
        def __init__(self):
            self.requests = []

        def plan(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise RecoverablePlanningError("Completion content is not valid JSON")
            return PlannerResult(
                PlanStatus.PLAN,
                (SkillAction("PICK", {"object_id": "mug"}),),
            )

    planner = RetryPlanner()
    states = iter((_state(1, ("mug",)), _state(2, ("mug",)), _state(3, ("mug",))))
    executive = DiscoveryReplanningExecutive(
        scene="kitchen",
        goal="test",
        observer=lambda: next(states),
        planner=planner,
        dispatcher=FakeDispatcher([SkillResult.succeeded("picked(mug)")]),
        goal_verifier=lambda _goal, state, _history: state.revision >= 3,
    )

    _run(executive)

    assert executive.mode == "complete"
    assert executive.replans == 1
    assert planner.requests[1].replan_event is not None
    assert planner.requests[1].replan_event.event_id == "invalid_planner_output"


def test_pre_action_failure_prevents_physical_dispatch_and_requests_replan():
    dispatcher = FakeDispatcher([])
    executive = DiscoveryReplanningExecutive(
        scene="kitchen",
        goal="test",
        observer=lambda: _state(1, ("mug",)),
        planner=ScriptedPlanner(
            [PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "mug"}),))]
        ),
        dispatcher=dispatcher,
        goal_verifier=lambda *_args: False,
        pre_action_check=lambda _action, _state: SkillResult.failed(
            FailureCode.OBJECT_NOT_VISIBLE, "object left the camera view"
        ),
        max_replans=0,
    )

    _run(executive)

    assert executive.mode == "failed"
    assert dispatcher.started == []
    assert executive.executed_actions == 0
