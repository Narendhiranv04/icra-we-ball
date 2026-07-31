import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mujoco_scenes.sequential_inspection import (
    DIRECT_ACTUATION_STEPS,
    SequentialInspectionAdapter,
    resolve_grounding_mode,
    run_fixed_order_inspection,
)


class FakeScene:
    def __init__(self, regions):
        self.states = {
            region: {"open": False, "inspected": False}
            for region in regions
        }

    def get_region_observation_states(self):
        return {
            region: {"region_id": region, **state}
            for region, state in self.states.items()
        }


class FakeAdapter:
    def __init__(self, scene):
        self.scene = scene
        self.opened = []

    def inspect(self, region):
        self.opened.append(region)
        self.scene.states[region] = {"open": True, "inspected": True}


class FakeSession:
    def __init__(self, statuses, root):
        self.statuses = iter(statuses)
        self.latest_witness = None
        self.events = []
        self.registry = {"objects": {}, "current_stage": -1}
        self.next_stage = 0
        self.root = Path(root)

    def append_event(self, event):
        self.events.append(dict(event))

    def observe(self, label, region):
        status = next(self.statuses)
        self.latest_witness = {
            "task_id": "synthetic",
            "stage": self.next_stage,
            "status": status,
        }
        self.registry["current_stage"] = self.next_stage
        stage = self.root / f"{self.next_stage:03d}_{label}"
        stage.mkdir()
        self.next_stage += 1
        return SimpleNamespace(total_points=0), stage


class SequentialWitnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.scene = FakeScene(("D1", "D2", "C2"))
        self.adapter = FakeAdapter(self.scene)

    def tearDown(self):
        self.temporary.cleanup()

    def run_loop(self, statuses):
        session = FakeSession(statuses, self.temporary.name)
        run_fixed_order_inspection(
            self.scene,
            session,
            ("D1", "D2", "C2"),
            adapter=self.adapter,
            observe=session.observe,
            stop_on_complete=True,
        )
        return session

    def test_initial_complete_opens_no_regions(self):
        session = self.run_loop(("COMPLETE",))
        self.assertEqual(self.adapter.opened, [])
        self.assertEqual(
            session.events[-1]["event"],
            "INSPECTION_STOPPED_COMPLETE",
        )

    def test_initial_noncomplete_opens_first_and_stops_on_complete(self):
        session = self.run_loop(("INCOMPLETE", "COMPLETE"))
        self.assertEqual(self.adapter.opened, ["D1"])
        self.assertFalse(self.scene.states["D2"]["inspected"])
        self.assertFalse(self.scene.states["C2"]["inspected"])
        self.assertEqual(
            session.events[-1]["remaining_regions"], ["D2", "C2"]
        )

    def test_indeterminate_continues_to_next_fixed_region(self):
        self.run_loop(("INDETERMINATE", "INDETERMINATE", "COMPLETE"))
        self.assertEqual(self.adapter.opened, ["D1", "D2"])
        self.assertFalse(self.scene.states["C2"]["inspected"])

    def test_exhaustion_is_clean_and_recorded(self):
        session = self.run_loop(
            ("INCOMPLETE", "INCOMPLETE", "INDETERMINATE", "INCOMPLETE")
        )
        self.assertEqual(self.adapter.opened, ["D1", "D2", "C2"])
        self.assertEqual(
            session.events[-1],
            {
                "event": "INSPECTION_ORDER_EXHAUSTED",
                "final_witness_status": "INCOMPLETE",
            },
        )

    def test_without_stop_flag_preserves_full_sequence(self):
        session = FakeSession(
            ("COMPLETE", "COMPLETE", "COMPLETE", "COMPLETE"),
            self.temporary.name,
        )
        run_fixed_order_inspection(
            self.scene,
            session,
            ("D1", "D2", "C2"),
            adapter=self.adapter,
            observe=session.observe,
            stop_on_complete=False,
        )
        self.assertEqual(self.adapter.opened, ["D1", "D2", "C2"])
        self.assertFalse(
            any(
                event["event"] == "INSPECTION_STOPPED_COMPLETE"
                for event in session.events
            )
        )

    def test_no_robot_adapter_uses_bounded_direct_actuation(self):
        class DirectScene:
            def __init__(self):
                self.state = SimpleNamespace(
                    container_open_state={"C2": False, "B1": False}
                )
                self.calls = []

            def open_container(self, region_id, *, steps):
                self.calls.append(("open", region_id, steps))

            def close_container(self, region_id, *, steps):
                self.calls.append(("close", region_id, steps))

        scene = DirectScene()
        adapter = SequentialInspectionAdapter(scene)
        adapter.inspect("C2")
        self.assertEqual(
            scene.calls,
            [("open", "C2", DIRECT_ACTUATION_STEPS)],
        )

        scene.state.container_open_state["C2"] = True
        adapter.inspect("B1")
        self.assertEqual(
            scene.calls[-2:],
            [
                ("close", "C2", DIRECT_ACTUATION_STEPS),
                ("open", "B1", DIRECT_ACTUATION_STEPS),
            ],
        )

    def test_auto_mode_uses_joint_only_for_joint_role_tasks(self):
        self.assertEqual(
            resolve_grounding_mode(
                {"_task_schema": "JOINT_ROLE_GROUNDING"}, "auto"
            ),
            "joint",
        )
        self.assertEqual(
            resolve_grounding_mode(
                {"_task_schema": "GEOMETRY_ONLY"}, "auto"
            ),
            "geometry-only",
        )
        self.assertEqual(
            resolve_grounding_mode(
                {"_task_schema": "JOINT_ROLE_GROUNDING"},
                "semantic-only",
            ),
            "semantic-only",
        )


if __name__ == "__main__":
    unittest.main()
