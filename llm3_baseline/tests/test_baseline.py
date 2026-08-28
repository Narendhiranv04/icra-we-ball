from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.inference import PlanningError, validate_images
from llm3_baseline.catalog import (
    load_catalog,
    load_parameter_catalog,
    scene_actions,
    scene_parameters,
)
from llm3_baseline.executive import LLM3Executive, ObservationFrame
from llm3_baseline.models import (
    Action,
    ActionResult,
    Entity,
    Observation,
    Plan,
    Region,
    ValidationError,
    parse_llm3_plan,
    parse_plan,
)
from llm3_baseline.planner import LLM3Planner, PlanResult, PlannerConfig
from llm3_baseline.prompt import SYSTEM_PROMPT, response_schema
from llm3_baseline.run_kitchen import _print_plan


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

    def test_observation_rejects_boolean_revision(self):
        payload = observation().as_prompt_dict()
        payload["revision"] = True
        with self.assertRaisesRegex(ValidationError, "revision"):
            Observation.from_dict(payload)

    def test_shared_image_validation_checks_types_and_base64(self):
        with self.assertRaisesRegex(ValueError, "camera must be a string"):
            validate_images(({"camera": 3, "data_url": IMAGE["data_url"]},))
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            validate_images(({"camera": "front", "data_url": "data:image/png;base64,!"},))

    def test_parameter_catalogue_rejects_invalid_bounds(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scenes": {
                            "kitchen": {
                                "PICK": {
                                    "offset": {"minimum": 1.0, "maximum": 0.0}
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid bounds"):
                load_parameter_catalog(path)

    def test_run_directory_refuses_to_mix_episode_artifacts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "episode"
            self.assertEqual(prepare_run_directory(output), output.resolve())
            (output / "model_call.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare_run_directory(output)

    def test_json_artifact_writer_replaces_complete_document(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            write_json(output, {"revision": 1})
            write_json(output, {"revision": 2})

            self.assertEqual(json.loads(output.read_text()), {"revision": 2})
            self.assertFalse((output.parent / ".trace.json.tmp").exists())

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

    def test_continuous_parameters_are_required_and_range_checked(self):
        parameters = scene_parameters(load_parameter_catalog(), "kitchen")
        valid = parse_llm3_plan(
            {
                "status": "PLAN",
                "reasoning": "Sample a collision-free serving placement.",
                "actions": [
                    {
                        "skill": "PLACE",
                        "arguments": {
                            "object_id": "mug_1",
                            "region_id": "D1",
                        },
                        "parameters": {
                            "x_offset_m": 0.01,
                            "y_offset_m": -0.01,
                            "yaw_rad": 0.5,
                        },
                    }
                ],
            },
            self.actions,
            parameters,
            observation(),
            max_actions=10,
        )
        self.assertEqual(valid.actions[0].parameters["yaw_rad"], 0.5)
        invalid = valid.as_dict()
        invalid["actions"][0]["parameters"]["x_offset_m"] = 0.5
        with self.assertRaisesRegex(ValidationError, "outside"):
            parse_llm3_plan(
                invalid,
                self.actions,
                parameters,
                observation(),
                max_actions=10,
            )

    def test_prompt_does_not_force_inspection(self):
        self.assertNotIn("return PLAN with an INSPECT", SYSTEM_PROMPT)
        self.assertNotIn("reason to inspect", SYSTEM_PROMPT)

    def test_schema_keeps_plan_and_no_plan_cardinality_exclusive(self):
        parameters = scene_parameters(load_parameter_catalog(), "kitchen")
        schema = response_schema(self.actions, parameters, 20)
        cardinality = [
            (
                row["properties"]["status"]["const"],
                row["properties"]["actions"]["minItems"],
                row["properties"]["actions"]["maxItems"],
            )
            for row in schema["oneOf"]
        ]
        self.assertEqual(cardinality, [("PLAN", 1, 20), ("NO_VALID_PLAN", 0, 0)])

    def test_planner_does_not_ask_model_when_goal_verifier_is_true(self):
        transport = FakeTransport({})
        planner = LLM3Planner(
            PlannerConfig(model="test-model"), transport=transport
        )

        result = planner.plan("Done goal", observation(goal=True), (IMAGE,))

        self.assertEqual(result.plan.status, "GOAL_COMPLETE")
        self.assertIsNone(transport.payload)

    def test_model_cannot_claim_goal_completion(self):
        parameters = scene_parameters(load_parameter_catalog(), "kitchen")
        with self.assertRaisesRegex(ValidationError, "independent verifier"):
            parse_llm3_plan(
                {
                    "status": "GOAL_COMPLETE",
                    "reasoning": "Looks done.",
                    "actions": [],
                },
                self.actions,
                parameters,
                observation(),
                max_actions=10,
            )

    def test_terminal_plan_is_rendered_as_simple_action_lines(self):
        stream = StringIO()
        with redirect_stdout(stream):
            _print_plan(
                2,
                Plan(
                    "PLAN",
                    (Action("PICK", {"object_id": "mug_1"}),),
                ),
            )
        self.assertEqual(
            stream.getvalue().splitlines(),
            ["[llm3 plan 2] PLAN", "  1. PICK object_id=mug_1"],
        )

    def test_planner_sends_images_visible_state_and_strict_schema(self):
        transport = FakeTransport(
            {
                "status": "PLAN",
                "reasoning": "The mug is visible and the gripper is empty.",
                "actions": [
                    {
                        "skill": "PICK",
                        "arguments": {"object_id": "mug_1"},
                        "parameters": {},
                    }
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
        state = task["textualized_state"]
        self.assertEqual(state["visible_objects"][0]["id"], "mug_1")
        self.assertNotIn("label", state["visible_objects"][0])
        self.assertNotIn("semantic_label", json.dumps(state))
        self.assertNotIn("observation", task)
        self.assertIn("continuous_parameter_catalog", task)

    def test_model_profile_supplies_recommended_thinking_settings(self):
        config = PlannerConfig.from_env({"LLM3_PROFILE": "qwen35-9b"})
        self.assertEqual(config.model, "qwen35-9b")
        self.assertEqual(config.max_tokens, 24576)
        self.assertEqual(config.sampling["temperature"], 1.0)
        self.assertEqual(config.sampling["top_k"], 20)
        self.assertTrue(config.toggle_thinking)

    def test_prompt_constrained_profile_omits_response_format(self):
        config = PlannerConfig.from_env({"LLM3_PROFILE": "internvl35-14b"})
        transport = FakeTransport(
            {
                "status": "PLAN",
                "reasoning": "The mug is visible and the gripper is empty.",
                "actions": [
                    {
                        "skill": "PICK",
                        "arguments": {"object_id": "mug_1"},
                        "parameters": {},
                    }
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

    def test_planner_config_rejects_boolean_numeric_limits(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            PlannerConfig(model="test", max_tokens=True)

    def test_planner_config_rejects_non_integer_action_limit(self):
        with self.assertRaisesRegex(ValueError, "max_actions"):
            PlannerConfig(model="test", max_actions=2.5)


class FakePlanner:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.failures = []

    def plan(self, goal, observation, images, *, history=(), failure=None):
        self.failures.append(failure)
        return PlanResult(next(self.plans), "fake", 0.0)


class InvalidThenValidPlanner(FakePlanner):
    def __init__(self, plan):
        super().__init__([plan])
        self.calls = 0

    def plan(self, *args, **kwargs):
        self.failures.append(kwargs.get("failure"))
        self.calls += 1
        if self.calls == 1:
            raise PlanningError("Invalid model plan: malformed JSON")
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
    def test_post_action_verification_does_not_recapture_images(self):
        world = FakeWorld()
        frame_calls = 0
        state_calls = 0

        def observe_frame():
            nonlocal frame_calls
            frame_calls += 1
            return world.observe()

        def observe_state():
            nonlocal state_calls
            state_calls += 1
            return world.state

        planner = FakePlanner(
            [Plan("PLAN", (Action("PICK", {"object_id": "mug_1"}),))]
        )
        result = LLM3Executive(
            planner,
            observe_frame,
            world,
            state_observer=observe_state,
        ).run("Pick the mug")

        self.assertTrue(result.success)
        self.assertEqual(frame_calls, 1)
        self.assertEqual(state_calls, 1)

    def test_plan_preparation_failure_is_reprompted(self):
        class PreparationWorld(FakeWorld):
            def __init__(self):
                super().__init__()
                self.preparations = 0

            def prepare(self, _actions):
                self.preparations += 1
                if self.preparations == 1:
                    return ActionResult.failed("collision", "Plan intersects cabinet")
                return ActionResult.succeeded()

        world = PreparationWorld()
        pick = Action("PICK", {"object_id": "mug_1"})
        planner = FakePlanner([Plan("PLAN", (pick,)), Plan("PLAN", (pick,))])

        result = LLM3Executive(
            planner, world.observe, world, max_model_calls=2
        ).run("Pick the mug")

        self.assertTrue(result.success)
        self.assertEqual(result.executed_actions, 1)
        self.assertEqual(planner.failures[1].code, "collision")

    def test_plan_preparation_exception_is_terminal_and_structured(self):
        class BrokenPreparationWorld(FakeWorld):
            def prepare(self, _actions):
                raise RuntimeError("authorization backend failed")

        world = BrokenPreparationWorld()
        planner = FakePlanner(
            [Plan("PLAN", (Action("PICK", {"object_id": "mug_1"}),))]
        )

        result = LLM3Executive(planner, world.observe, world).run("Pick the mug")

        self.assertFalse(result.success)
        self.assertEqual(result.status, "NON_RECOVERABLE_FAILURE")
        self.assertEqual(result.terminal_failure.code, "internal_error")

    def test_invalid_model_output_is_reprompted_without_motion(self):
        world = FakeWorld()
        planner = InvalidThenValidPlanner(
            Plan("PLAN", (Action("PICK", {"object_id": "mug_1"}),))
        )

        result = LLM3Executive(
            planner, world.observe, world, max_model_calls=2
        ).run("Pick the mug")

        self.assertTrue(result.success)
        self.assertEqual(result.executed_actions, 1)
        self.assertEqual(planner.failures[1].code, "invalid_model_output")
        self.assertIsNone(result.planning_trace[0]["full_plan"])

    def test_invalid_hand_state_sequence_is_reprompted_before_motion(self):
        world = FakeWorld()
        invalid = Plan(
            "PLAN",
            (
                Action("PICK", {"object_id": "mug_1"}),
                Action(
                    "STIR",
                    {"tool_id": "spoon_1", "target_id": "mug_1"},
                ),
            ),
        )
        planner = FakePlanner(
            [
                invalid,
                Plan(
                    "PLAN", (Action("PICK", {"object_id": "mug_1"}),)
                ),
            ]
        )
        result = LLM3Executive(
            planner, world.observe, world, max_model_calls=2
        ).run("Pick the mug")

        self.assertTrue(result.success)
        self.assertEqual(result.executed_actions, 1)
        self.assertEqual(planner.failures[1].code, "precondition_failed")
        self.assertIn("PICK spoon_1 first", planner.failures[1].message)

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

    def test_no_valid_plan_is_not_overridden_by_uninspected_regions(self):
        world = FakeWorld()
        planner = FakePlanner([Plan("NO_VALID_PLAN", ())])
        result = LLM3Executive(
            planner, world.observe, world, max_model_calls=3
        ).run("Find and pick a spoon")
        self.assertFalse(result.success)
        self.assertEqual(result.status, "NO_VALID_PLAN")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.executed_actions, 0)

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
