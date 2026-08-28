from mujoco_scenes.tamp.grounded_execution import (
    ExecutionFailure,
    GroundedPlanExecutive,
    GroundedTask,
)
from mujoco_scenes.tamp.skills import FailureCode, SkillAction, SkillResult
from mujoco_scenes.tamp.state import ObservedState, RobotObservation


class World:
    def __init__(self):
        self.revision = 0
        self.complete = False
        self.results = []
        self.action = None

    def observe(self):
        self.revision += 1
        return ObservedState({}, {}, RobotObservation("home"), revision=self.revision)

    def start(self, action):
        self.action = action

    def update(self):
        result = self.results.pop(0) if self.results else SkillResult.succeeded()
        if result.success and self.action.name == "place":
            self.complete = True
        return result


class Sequencer:
    def __init__(self):
        self.failures = []

    def sequence(self, _task, _state, *, history, failure):
        self.failures.append(failure)
        return (SkillAction("place", {"object": "mug", "region": "serving"}),)


def _run(executive):
    for _ in range(20):
        executive.update()
        if not executive.busy:
            return
    raise AssertionError("executive did not terminate")


def test_primary_execution_uses_verified_handoff_and_observed_goal():
    world = World()
    sequencer = Sequencer()
    executive = GroundedPlanExecutive(
        world.observe,
        sequencer,
        world,
        lambda _task, _state, _history: world.complete,
    )
    task = GroundedTask("serve", "Serve coffee", {"status": "COMPLETE"})
    executive.start(task)
    _run(executive)
    assert executive.mode == "complete"
    assert executive.executed_actions == 1
    assert sequencer.failures == [None]


def test_recoverable_failure_returns_to_deterministic_sequencer_not_fm():
    world = World()
    world.results = [
        SkillResult.failed(FailureCode.IK_FAILED, "No IK"),
        SkillResult.succeeded(),
    ]
    sequencer = Sequencer()
    executive = GroundedPlanExecutive(
        world.observe,
        sequencer,
        world,
        lambda _task, _state, _history: world.complete,
        max_replans=1,
    )
    executive.start(GroundedTask("serve", "Serve coffee", {"status": "COMPLETE"}))
    _run(executive)
    assert executive.mode == "complete"
    assert isinstance(sequencer.failures[1], ExecutionFailure)
    assert sequencer.failures[1].code is FailureCode.IK_FAILED


def test_nonrecoverable_failure_stops_without_replanning():
    world = World()
    world.results = [
        SkillResult.failed(
            FailureCode.INTERNAL_ERROR, "broken", recoverable=False
        )
    ]
    sequencer = Sequencer()
    executive = GroundedPlanExecutive(
        world.observe,
        sequencer,
        world,
        lambda _task, _state, _history: False,
    )
    executive.start(GroundedTask("serve", "Serve coffee", {"status": "COMPLETE"}))
    _run(executive)
    assert executive.mode == "failed"
    assert len(sequencer.failures) == 1


def test_plan_preparation_failure_uses_bounded_replanning():
    class PreparationWorld(World):
        def __init__(self):
            super().__init__()
            self.preparations = 0

        def prepare(self, _actions):
            self.preparations += 1
            if self.preparations == 1:
                return SkillResult.failed(FailureCode.COLLISION, "Plan collision")
            return SkillResult.succeeded()

    world = PreparationWorld()
    sequencer = Sequencer()
    executive = GroundedPlanExecutive(
        world.observe,
        sequencer,
        world,
        lambda _task, _state, _history: world.complete,
        max_replans=1,
    )

    executive.start(GroundedTask("serve", "Serve coffee", {"status": "COMPLETE"}))
    _run(executive)

    assert executive.mode == "complete"
    assert executive.replans == 1
    assert sequencer.failures[1].code is FailureCode.COLLISION


def test_status_names_the_action_before_synchronous_dispatch():
    world = World()
    observed_statuses = []
    original_start = world.start
    executive = None

    def start(action):
        observed_statuses.append(executive.status)
        original_start(action)

    world.start = start
    executive = GroundedPlanExecutive(
        world.observe,
        Sequencer(),
        world,
        lambda _task, _state, _history: world.complete,
    )
    executive.start(GroundedTask("serve", "Serve coffee", {"status": "COMPLETE"}))
    _run(executive)
    assert observed_statuses == ["Executing place (1/1)"]


def test_unexpected_dispatch_start_error_becomes_terminal_failure():
    world = World()

    def broken_start(_action):
        raise RuntimeError("backend crashed")

    world.start = broken_start
    executive = GroundedPlanExecutive(
        world.observe,
        Sequencer(),
        world,
        lambda _task, _state, _history: False,
    )

    executive.start(GroundedTask("serve", "Serve coffee", {}))
    _run(executive)

    assert executive.mode == "failed"
    assert executive.failure.code is FailureCode.INTERNAL_ERROR
    assert executive.executed_actions == 0


def test_initial_observation_error_becomes_terminal_failure():
    executive = GroundedPlanExecutive(
        lambda: (_ for _ in ()).throw(RuntimeError("camera unavailable")),
        Sequencer(),
        World(),
        lambda _task, _state, _history: False,
    )

    executive.start(GroundedTask("serve", "Serve coffee", {}))

    assert executive.mode == "failed"
    assert executive.failure.code is FailureCode.INTERNAL_ERROR
