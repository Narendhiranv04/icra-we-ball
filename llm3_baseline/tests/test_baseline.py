from __future__ import annotations

import json
import unittest

from llm3_baseline.catalog import load_catalog, scene_actions
from llm3_baseline.executive import LLM3Executive, ObservationFrame
from llm3_baseline.models import (
    Action,
    ActionResult,
    Entity,
    Observation,
    Plan,
    Region,
    ValidationError,
    parse_plan,
)
from llm3_baseline.planner import LLM3Planner, PlanResult, PlannerConfig


IMAGE = {"camera": "front", "data_url": "data:image/png;base64,AA=="}


def observation(*, entities=("mug_1",), goal=False, revision=0):
    return Observation(
        "kitchen",
        revision,
        tuple(Entity(item, "object", item) for item in entities),
        (Region("D1", "left drawer", "closed", False),),
        {"holding": None},
        goal,
    )


class FakeTransport:
    def __init__(self, content):
        self.content = content
        self.payload = None

    def complete(self, payload):
        self.payload = payload
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(self.content)},
                }
            ]
        }


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.actions = scene_actions(load_catalog(), "kitchen")

    def test_catalogue_covers_all_three_scenes(self):
        catalogue = load_catalog()
        self.assertEqual(
            set(catalogue["scenes"]), {"kitchen", "living_room", "workshop"}
        )
        self.assertIn("STIR", catalogue["scenes"]["kitchen"])
        self.assertIn("CLEAN", catalogue["scenes"]["living_room"])
        self.assertIn("FASTEN", catalogue["scenes"]["workshop"])

    def test_plan_rejects_unobserved_object(self):
        with self.assertRaisesRegex(ValidationError, "unobserved object"):
            parse_plan(
                {
                    "status": "PLAN",
                    "actions": [
                        {
                            "skill": "PICK",
                            "arguments": {"object_id": "hidden_spoon"},
                        }
                    ],
                },
                self.actions,
                observation(),
                max_actions=10,
            )

    def test_plan_allows_inspection_of_known_region(self):
        plan = parse_plan(
            {
                "status": "PLAN",
                "actions": [
                    {"skill": "INSPECT", "arguments": {"region_id": "D1"}}
                ],
            },
            self.actions,
            observation(),
            max_actions=10,
        )
        self.assertEqual(plan.actions[0].arguments["region_id"], "D1")

    def test_planner_sends_images_visible_state_and_strict_schema(self):
        transport = FakeTransport(
            {
                "status": "PLAN",
                "actions": [
                    {"skill": "PICK", "arguments": {"object_id": "mug_1"}}
                ],
            }
        )
        planner = LLM3Planner(
            PlannerConfig(model="test-model"), transport=transport
        )
        result = planner.plan("Pick the mug", observation(), (IMAGE,))
        self.assertEqual(result.plan.actions[0].skill, "PICK")
        payload = transport.payload
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(
            payload["response_format"]["json_schema"]["name"],
            "llm3_action_plan",
        )
        task = json.loads(payload["messages"][1]["content"][0]["text"])
        self.assertEqual(task["observation"]["visible_entities"][0]["id"], "mug_1")
        self.assertNotIn("hidden_entities", task["observation"])

    def test_model_profile_supplies_recommended_thinking_settings(self):
        config = PlannerConfig.from_env({"LLM3_PROFILE": "qwen35-9b"})
        self.assertEqual(config.model, "qwen35-9b")
        self.assertEqual(config.max_tokens, 12288)
        self.assertEqual(config.sampling["temperature"], 1.0)
        self.assertEqual(config.sampling["top_k"], 20)
        self.assertTrue(config.toggle_thinking)

    def test_prompt_constrained_profile_omits_response_format(self):
        config = PlannerConfig.from_env({"LLM3_PROFILE": "internvl35-14b"})
        transport = FakeTransport(
            {
                "status": "PLAN",
                "actions": [
                    {"skill": "PICK", "arguments": {"object_id": "mug_1"}}
                ],
            }
        )
        LLM3Planner(config, transport=transport).plan(
            "Pick the mug", observation(), (IMAGE,)
        )
        self.assertNotIn("response_format", transport.payload)
        task = json.loads(transport.payload["messages"][1]["content"][0]["text"])
        self.assertIn("output_schema", task)

    def test_fixed_thinking_profile_rejects_disable_override(self):
        with self.assertRaisesRegex(ValueError, "fixed-thinking"):
            PlannerConfig.from_env(
                {
                    "LLM3_PROFILE": "qwen3-vl-8b-thinking",
                    "LLM3_ENABLE_THINKING": "false",
                }
            )


class FakePlanner:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.failures = []

    def plan(self, goal, observation, images, *, history=(), failure=None):
        self.failures.append(failure)
        return PlanResult(next(self.plans), "fake", 0.0)


class FakeWorld:
    def __init__(self):
        self.state = observation()
        self.results = []

    def observe(self):
        return ObservationFrame(self.state, (IMAGE,))

    def execute(self, action):
        if self.results:
            result = self.results.pop(0)
        else:
            result = ActionResult.succeeded()
        if result.success and action.skill == "INSPECT":
            self.state = observation(
                entities=("mug_1", "spoon_1"), revision=1
            )
        if result.success and action.skill == "PICK":
            self.state = observation(
                entities=tuple(item.entity_id for item in self.state.entities),
                goal=True,
                revision=self.state.revision + 1,
            )
        return result


class ExecutiveTests(unittest.TestCase):
    def test_replans_after_motion_failure(self):
        world = FakeWorld()
        world.results = [
            ActionResult.failed("ik_failed", "No collision-free IK."),
            ActionResult.succeeded(),
        ]
        pick = Action("PICK", {"object_id": "mug_1"})
        planner = FakePlanner(
            [Plan("PLAN", (pick,)), Plan("PLAN", (pick,))]
        )
        result = LLM3Executive(
            planner, world.observe, world, max_model_calls=3
        ).run("Pick the mug")
        self.assertTrue(result.success)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(planner.failures[1].code, "ik_failed")

    def test_inspection_forces_fresh_observation_and_replan(self):
        world = FakeWorld()
        planner = FakePlanner(
            [
                Plan("PLAN", (Action("INSPECT", {"region_id": "D1"}),)),
                Plan("PLAN", (Action("PICK", {"object_id": "spoon_1"}),)),
            ]
        )
        result = LLM3Executive(planner, world.observe, world).run(
            "Find and pick a spoon"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.executed_actions, 2)

    def test_nonrecoverable_failure_stops_without_replanning(self):
        world = FakeWorld()
        world.results = [
            ActionResult.failed(
                "internal_error", "Executor unavailable.", recoverable=False
            )
        ]
        planner = FakePlanner(
            [Plan("PLAN", (Action("PICK", {"object_id": "mug_1"}),))]
        )
        result = LLM3Executive(planner, world.observe, world).run("Pick mug")
        self.assertFalse(result.success)
        self.assertEqual(result.status, "NON_RECOVERABLE_FAILURE")
        self.assertEqual(result.model_calls, 1)


if __name__ == "__main__":
    unittest.main()
