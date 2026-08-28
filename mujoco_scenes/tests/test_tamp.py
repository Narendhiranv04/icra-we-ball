import json
import tempfile
import time
import unittest
from pathlib import Path

from mujoco_scenes.foundation_model import (
    FixedAssessmentBackend,
    FunctionalAssessmentResult,
)
from mujoco_scenes.tamp.candidates import visible_candidates
from mujoco_scenes.tamp.events import EventLog
from mujoco_scenes.tamp.executive import FunctionalTask, TaskExecutive
from mujoco_scenes.tamp.functions import FunctionRegistry
from mujoco_scenes.tamp.skills import (
    FailureCode,
    SkillAction,
    SkillResult,
)
from mujoco_scenes.tamp.state import (
    ObjectObservation,
    ObservedState,
    RegionObservation,
    RobotObservation,
)


def _state(*, visible_regions=("right_drawer", "left_drawer"), location="home"):
    regions = {
        region_id: RegionObservation(
            region_id,
            "drawer",
            region_id in visible_regions,
            inspected=True,
            open=True,
            occupied_by=(),
        )
        for region_id in ("right_drawer", "left_drawer")
    }
    return ObservedState(
        objects={
            "controller": ObjectObservation(
                "controller",
                "game_controller",
                True,
                location,
                {"rigid": True},
            )
        },
        regions=regions,
        robot=RobotObservation("home"),
    )


class MutableObserver:
    def __init__(self, state):
        self.state = state

    def __call__(self):
        return self.state


class FakeDispatcher:
    def __init__(self, observer, failures=None):
        self.observer = observer
        self.failures = dict(failures or {})
        self.action = None

    def start(self, action):
        self.action = action

    def update(self):
        action = self.action
        self.action = None
        candidate = action.arguments.get("candidate")
        failure = self.failures.pop(candidate, None)
        if failure is not None:
            return SkillResult.failed(failure, f"{candidate} failed")
        if action.name == "discover":
            self.observer.state = _state(visible_regions=("left_drawer",))
        elif action.name == "place":
            self.observer.state = _state(location=str(candidate))
        return SkillResult.succeeded()


def _run(executive):
    for _ in range(1000):
        executive.update()
        if not executive.busy:
            return
        time.sleep(0.0005)
    raise AssertionError("executive did not terminate")


class TampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = FunctionRegistry.load()
        cls.task = FunctionalTask(
            "store_controller",
            "controller",
            "can_store",
            "stored_in",
        )

    def test_candidate_generation_excludes_unseen_regions(self):
        state = _state(visible_regions=("right_drawer",))

        candidates = visible_candidates(
            state, self.registry.get("can_store")
        )

        self.assertEqual(
            tuple(candidate.candidate_id for candidate in candidates),
            ("right_drawer",),
        )

    def test_candidate_generation_excludes_internal_planner_facts(self):
        state = _state(visible_regions=("right_drawer",))
        region = state.regions["right_drawer"]
        state = ObservedState(
            state.objects,
            {
                **state.regions,
                "right_drawer": RegionObservation(
                    region.region_id,
                    region.category,
                    region.visible,
                    inspected=region.inspected,
                    open=region.open,
                    occupied_by=region.occupied_by,
                    facts={"rigid": True, "_place_site": "internal_site"},
                ),
            },
            state.robot,
        )

        candidates = visible_candidates(
            state, self.registry.get("can_store")
        )

        self.assertEqual(candidates[0].facts["rigid"], True)
        self.assertNotIn("_place_site", candidates[0].facts)

    def test_executive_falls_back_after_recoverable_failure(self):
        observer = MutableObserver(_state())
        dispatcher = FakeDispatcher(
            observer,
            failures={"right_drawer": FailureCode.PLACEMENT_FAILED},
        )

        def plan(_task, candidate, _state):
            return (
                SkillAction(
                    "place", {"candidate": candidate.candidate_id}
                ),
            )

        def verify(_task, candidate, state):
            return (
                state.objects["controller"].location
                == candidate.candidate_id
            )

        executive = TaskExecutive(
            self.registry,
            FixedAssessmentBackend(
                ("right_drawer", "left_drawer"),
                ("right_drawer", "left_drawer"),
            ),
            observer,
            dispatcher,
            plan,
            verify,
        )
        try:
            executive.start(self.task)
            _run(executive)
        finally:
            executive.close()

        self.assertEqual(executive.mode, "complete")
        self.assertIn("left_drawer", executive.status)
        failed = [
            event
            for event in executive.events.events
            if event["event"] == "candidate_failed"
        ]
        self.assertEqual(failed[0]["candidate"], "right_drawer")

    def test_discovery_reobserves_before_assessment(self):
        observer = MutableObserver(_state(visible_regions=()))
        dispatcher = FakeDispatcher(observer)

        def plan(_task, candidate, _state):
            return (
                SkillAction(
                    "place", {"candidate": candidate.candidate_id}
                ),
            )

        executive = TaskExecutive(
            self.registry,
            FixedAssessmentBackend(("left_drawer",)),
            observer,
            dispatcher,
            plan,
            lambda _task, candidate, state: (
                state.objects["controller"].location
                == candidate.candidate_id
            ),
            discovery_policy=lambda _task, _state, _attempt: (
                SkillAction("discover"),
            ),
            max_discoveries=1,
        )
        try:
            executive.start(self.task)
            _run(executive)
        finally:
            executive.close()

        self.assertEqual(executive.mode, "complete")
        self.assertTrue(
            any(
                event["event"] == "discovery_complete"
                for event in executive.events.events
            )
        )

    def test_jsonl_event_log_is_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with EventLog(path) as events:
                events.append("test", visible_ids=["controller"])

            record = json.loads(path.read_text().strip())

        self.assertEqual(record["event"], "test")
        self.assertEqual(record["visible_ids"], ["controller"])

    def test_executive_reobserves_after_each_skill(self):
        observer = MutableObserver(_state())
        observations = 0

        def counted_observer():
            nonlocal observations
            observations += 1
            return observer()

        def plan(_task, candidate, _state):
            return (
                SkillAction(
                    "place", {"candidate": candidate.candidate_id}
                ),
            )

        executive = TaskExecutive(
            self.registry,
            FixedAssessmentBackend(("right_drawer",)),
            counted_observer,
            FakeDispatcher(observer),
            plan,
            lambda _task, candidate, state: (
                state.objects["controller"].location
                == candidate.candidate_id
            ),
        )
        try:
            executive.start(self.task)
            _run(executive)
        finally:
            executive.close()

        self.assertGreaterEqual(observations, 3)
        self.assertTrue(
            any(
                event.get("after_skill") == "place"
                for event in executive.events.events
            )
        )
        observation = next(
            event
            for event in executive.events.events
            if event.get("after_skill") == "place"
        )
        self.assertEqual(
            observation["state"]["objects"]["controller"]["location"],
            "right_drawer",
        )

    def test_unknown_custom_backend_candidate_fails_without_crashing(self):
        class InvalidBackend:
            def assess(self, _request):
                return FunctionalAssessmentResult(
                    ("invented",), ("invented",), "invalid"
                )

        observer = MutableObserver(_state())
        executive = TaskExecutive(
            self.registry,
            InvalidBackend(),
            observer,
            FakeDispatcher(observer),
            lambda _task, _candidate, _state: (),
            lambda _task, _candidate, _state: False,
        )
        try:
            executive.start(self.task)
            _run(executive)
        finally:
            executive.close()

        self.assertEqual(executive.mode, "failed")
        self.assertEqual(executive.failure_code, FailureCode.INFERENCE_FAILED)

    def test_negative_discovery_limit_is_rejected(self):
        observer = MutableObserver(_state())
        with self.assertRaisesRegex(ValueError, "max_discoveries"):
            TaskExecutive(
                self.registry,
                FixedAssessmentBackend(("right_drawer",)),
                observer,
                FakeDispatcher(observer),
                lambda _task, _candidate, _state: (),
                lambda _task, _candidate, _state: False,
                max_discoveries=-1,
            )


if __name__ == "__main__":
    unittest.main()
