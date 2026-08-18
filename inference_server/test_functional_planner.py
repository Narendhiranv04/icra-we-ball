import json
import socket
import threading
import unittest
import urllib.error
import urllib.request

from inference_server.functional_planner import (
    FunctionalPlanner,
    FunctionalPlanValidationError,
    PlannerConfig,
    SYSTEM_PROMPT,
    completion_payload,
    load_function_catalog,
    validate_decomposition,
)
from inference_server.planner_api import build_server, config_from_env


IMAGE = "data:image/png;base64,AA=="


def local_sockets_available():
    try:
        probe = socket.socket()
    except PermissionError:
        return False
    probe.close()
    return True


def request(scene="kitchen"):
    return {
        "scene": scene,
        "goal": "Make and stir coffee." if scene == "kitchen" else "Clean the table.",
        "images": [{"camera": "front", "data_url": IMAGE}],
    }


def decomposition():
    return {
        "status": "DECOMPOSED",
        "scene": "kitchen",
        "goal_summary": "Prepare coffee in a suitable vessel and stir it.",
        "functional_requirements": [
            {
                "id": "req_1",
                "function": "can_hold_liquid",
                "candidate_kind": "object",
                "purpose": "Contain the prepared coffee.",
                "target_description": "The prepared coffee.",
                "ranked_candidate_types": [
                    "coffee mug",
                    "ceramic cup",
                    "glass tumbler",
                    "travel mug",
                    "teacup",
                    "insulated cup",
                    "enamel cup",
                    "demitasse cup",
                    "measuring cup",
                    "drinking glass",
                ],
                "depends_on": [],
            },
            {
                "id": "req_2",
                "function": "can_stir",
                "candidate_kind": "object",
                "purpose": "Stir the coffee contents.",
                "target_description": "The selected liquid container.",
                "ranked_candidate_types": [
                    "teaspoon",
                    "coffee stirrer",
                    "chopstick",
                    "swizzle stick",
                    "small whisk",
                    "bar spoon",
                    "wooden stir stick",
                    "cocktail spoon",
                    "stirring rod",
                    "silicone spatula",
                ],
                "depends_on": ["req_1"],
            },
        ],
        "unsupported_reason": "",
    }


class FakeTransport:
    def __init__(self, result):
        self.result = result
        self.payload = None

    def complete(self, payload):
        self.payload = payload
        return {"choices": [{"message": {"content": json.dumps(self.result)}}]}


class FunctionalPlannerTests(unittest.TestCase):
    def setUp(self):
        self.config = PlannerConfig("http://model/v1", "vlm", "secret")

    def test_catalog_matches_repository_function_registry(self):
        catalog = load_function_catalog()
        self.assertEqual(catalog["min_ranked_candidates"], 10)
        self.assertEqual(catalog["max_ranked_candidates"], 15)
        self.assertEqual(
            set(catalog["functions"]),
            {"can_store", "can_stir", "can_hold_liquid", "can_clean", "can_spread"},
        )
        self.assertEqual(catalog["functions"]["can_store"]["candidate_kind"], "region")

    def test_payload_contains_images_functions_and_no_action_catalog(self):
        payload = completion_payload(request(), self.config)
        content = payload["messages"][1]["content"]
        task = json.loads(content[0]["text"])
        self.assertEqual(content[-1]["type"], "image_url")
        self.assertIn("can_stir", task["function_catalog"])
        self.assertNotIn("allowed_actions", task)
        self.assertEqual(task["minimum_ranking_length"], 10)
        self.assertEqual(task["ranking_limit"], 15)
        self.assertIn("utensil", task["forbidden_generic_candidate_types"])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["presence_penalty"], 1.5)

    def test_non_thinking_payload_uses_qwen_instruct_sampling(self):
        config = PlannerConfig(
            "http://model/v1",
            "vlm",
            "secret",
            enable_thinking=False,
        )
        payload = completion_payload(request(), config)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.8)

    def test_system_prompt_does_not_seed_candidate_types(self):
        prompt = SYSTEM_PROMPT.casefold()
        for candidate in (
            "spoon",
            "chopstick",
            "coffee stirrer",
            "mug",
            "side table",
        ):
            self.assertNotIn(candidate, prompt)

    def test_validates_ranked_functional_requirements(self):
        result = validate_decomposition(decomposition(), request())
        requirements = result["functional_requirements"]
        self.assertEqual(requirements[1]["ranked_candidate_types"][0], "teaspoon")

    def test_normalizes_empty_unsupported_reason_placeholder(self):
        result = decomposition()
        result["unsupported_reason"] = "N/A"
        validated = validate_decomposition(result, request())
        self.assertEqual(validated["unsupported_reason"], "")

    def test_reports_zero_decomposed_requirements_precisely(self):
        result = decomposition()
        result["functional_requirements"] = []
        with self.assertRaisesRegex(
            FunctionalPlanValidationError, "zero functional requirements"
        ):
            validate_decomposition(result, request())

    def test_accepts_supported_workshop_scene(self):
        unsupported = {
            "status": "GOAL_UNSUPPORTED",
            "scene": "workshop",
            "goal_summary": "Drive a screw.",
            "functional_requirements": [],
            "unsupported_reason": "No configured function represents screw driving.",
        }
        self.assertEqual(
            validate_decomposition(unsupported, request("workshop"))["status"],
            "GOAL_UNSUPPORTED",
        )

    def test_rejects_candidate_count_outside_ten_to_fifteen(self):
        result = decomposition()
        result["functional_requirements"][0]["ranked_candidate_types"] = ["mug"]
        with self.assertRaisesRegex(FunctionalPlanValidationError, "10-15"):
            validate_decomposition(result, request())
        result = decomposition()
        result["functional_requirements"][0]["ranked_candidate_types"].extend(
            ["water glass", "pint glass", "goblet", "beaker", "carafe", "pitcher"]
        )
        with self.assertRaisesRegex(FunctionalPlanValidationError, "10-15"):
            validate_decomposition(result, request())

    def test_rejects_generic_candidate_type(self):
        result = decomposition()
        result["functional_requirements"][1]["ranked_candidate_types"][1] = "utensil"
        with self.assertRaisesRegex(FunctionalPlanValidationError, "generic candidate"):
            validate_decomposition(result, request())
        result = decomposition()
        result["functional_requirements"][1]["ranked_candidate_types"][1] = (
            "cooking utensil"
        )
        with self.assertRaisesRegex(FunctionalPlanValidationError, "generic candidate"):
            validate_decomposition(result, request())

    def test_rejects_unknown_function_and_wrong_candidate_kind(self):
        result = decomposition()
        result["functional_requirements"][0]["function"] = "can_pour"
        with self.assertRaisesRegex(FunctionalPlanValidationError, "Unknown function"):
            validate_decomposition(result, request())
        result = decomposition()
        result["functional_requirements"][0]["candidate_kind"] = "region"
        with self.assertRaisesRegex(FunctionalPlanValidationError, "candidate_kind=object"):
            validate_decomposition(result, request())

    def test_rejects_dependency_cycle(self):
        result = decomposition()
        result["functional_requirements"][0]["depends_on"] = ["req_2"]
        with self.assertRaisesRegex(FunctionalPlanValidationError, "cycle"):
            validate_decomposition(result, request())

    def test_functional_planner_stops_before_search_or_execution(self):
        transport = FakeTransport(decomposition())
        result = FunctionalPlanner(self.config, transport).decompose(request())
        self.assertFalse(result["search_started"])
        self.assertFalse(result["semantic_grounding_complete"])
        self.assertFalse(result["geometry_verified"])
        self.assertFalse(result["execution_started"])
        self.assertIsNotNone(transport.payload)

    def test_reports_missing_final_content(self):
        class ReasoningOnlyTransport:
            def complete(self, payload):
                return {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": None,
                                "reasoning_content": "unfinished reasoning",
                            },
                        }
                    ]
                }

        with self.assertRaisesRegex(
            FunctionalPlanValidationError, "no final JSON content"
        ):
            FunctionalPlanner(self.config, ReasoningOnlyTransport()).decompose(request())

    def test_api_config_resolves_served_model_from_profile(self):
        config, incoming_key, host, port = config_from_env(
            {"INFERENCE_MODEL": "qwen35-9b", "INFERENCE_API_KEY": "secret"}
        )
        self.assertEqual(config.model, "qwen35-9b")
        self.assertTrue(config.enable_thinking)
        self.assertEqual(config.max_tokens, 12288)
        self.assertEqual(incoming_key, "secret")
        self.assertEqual((host, port), ("127.0.0.1", 8080))

    def test_api_config_allows_keyless_loopback_only(self):
        config, incoming_key, host, _port = config_from_env(
            {"INFERENCE_MODEL": "qwen35-9b", "PLANNER_HOST": "127.0.0.1"}
        )
        self.assertEqual(config.api_key, "")
        self.assertEqual(incoming_key, "")
        self.assertEqual(host, "127.0.0.1")
        with self.assertRaisesRegex(ValueError, "non-loopback"):
            config_from_env(
                {"INFERENCE_MODEL": "qwen35-9b", "PLANNER_HOST": "0.0.0.0"}
            )

    def test_api_config_can_disable_thinking(self):
        config, _key, _host, _port = config_from_env(
            {
                "INFERENCE_MODEL": "qwen35-9b",
                "PLANNER_ENABLE_THINKING": "false",
            }
        )
        self.assertFalse(config.enable_thinking)
        with self.assertRaisesRegex(ValueError, "must be true or false"):
            config_from_env(
                {
                    "INFERENCE_MODEL": "qwen35-9b",
                    "PLANNER_ENABLE_THINKING": "sometimes",
                }
            )

    def test_model_profiles_select_their_own_planner_settings(self):
        cases = {
            "glm46v-flash": (True, 12288, 0.8, True),
            "qwen3-vl-8b-thinking": (True, 12288, 1.0, True),
            "internvl35-14b": (True, 12288, 0.6, False),
            "kimi-vl-a3b-thinking": (True, 12288, 0.6, False),
        }
        for profile, expected in cases.items():
            with self.subTest(profile=profile):
                config, *_ = config_from_env({"INFERENCE_MODEL": profile})
                payload = completion_payload(request(), config)
                self.assertEqual(
                    (
                        config.enable_thinking,
                        config.max_tokens,
                        payload["temperature"],
                        "response_format" in payload,
                    ),
                    expected,
                )

    def test_thinking_sampling_matches_checkpoint_guidance(self):
        expected = {
            "qwen35-9b": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
            },
            "glm46v-flash": {
                "temperature": 0.8,
                "top_p": 0.6,
                "top_k": 2,
                "repetition_penalty": 1.1,
            },
            "qwen3-vl-8b-thinking": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
            "internvl35-14b": {
                "temperature": 0.6,
                "top_p": 0.95,
            },
            "kimi-vl-a3b-thinking": {"temperature": 0.6},
        }
        for profile, sampling in expected.items():
            with self.subTest(profile=profile):
                config, *_ = config_from_env({"INFERENCE_MODEL": profile})
                self.assertEqual(config.sampling, sampling)

    def test_fixed_thinking_profiles_reject_incompatible_override(self):
        with self.assertRaisesRegex(ValueError, "fixed-thinking"):
            config_from_env(
                {
                    "INFERENCE_MODEL": "qwen3-vl-8b-thinking",
                    "PLANNER_ENABLE_THINKING": "false",
                }
            )
        with self.assertRaisesRegex(ValueError, "fixed-thinking"):
            config_from_env(
                {
                    "INFERENCE_MODEL": "internvl35-14b",
                    "PLANNER_ENABLE_THINKING": "false",
                }
            )

    def test_internvl_uses_prompt_driven_thinking(self):
        config, *_ = config_from_env({"INFERENCE_MODEL": "internvl35-14b"})
        payload = completion_payload(request(), config)
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("<think>", system_prompt)
        self.assertNotIn("response_format", payload)
        self.assertEqual(config.reasoning_markers, ("<think>", "</think>"))

    def test_kimi_unconstrained_request_includes_schema_and_parses_final_json(self):
        config, *_ = config_from_env({"INFERENCE_MODEL": "kimi-vl-a3b-thinking"})
        payload = completion_payload(request(), config)
        task = json.loads(payload["messages"][1]["content"][0]["text"])
        self.assertIn("output_schema", task)
        transport = FakeTransport(decomposition())
        transport.complete = lambda _payload: {
            "choices": [
                {
                    "message": {
                        "content": "◁think▷hidden◁/think▷\n```json\n"
                        + json.dumps(decomposition())
                        + "\n```"
                    }
                }
            ]
        }
        result = FunctionalPlanner(config, transport).decompose(request())
        self.assertEqual(result["decomposition"]["status"], "DECOMPOSED")


@unittest.skipUnless(local_sockets_available(), "local sockets are sandboxed")
class FunctionalPlannerAPITests(unittest.TestCase):
    def setUp(self):
        config = PlannerConfig("http://model/v1", "vlm", "upstream")
        self.planner = FunctionalPlanner(config, FakeTransport(decomposition()))
        self.server = build_server("127.0.0.1", 0, self.planner, "client-secret")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def call(self, path, *, key=None, payload=None):
        headers = {}
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        query = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        with urllib.request.urlopen(query, timeout=2) as response:
            return response.status, json.load(response)

    def test_health_and_function_catalog(self):
        status, health = self.call("/health")
        self.assertEqual((status, health["service"]), (200, "functional-planner"))
        self.assertTrue(health["thinking_enabled"])
        self.assertTrue(health["structured_output"])
        self.assertEqual(health["max_tokens"], 8192)
        status, catalog = self.call("/v1/functions", key="client-secret")
        self.assertEqual(catalog["min_ranked_candidates"], 10)
        self.assertEqual(catalog["max_ranked_candidates"], 15)

    def test_decompose_endpoint_requires_authentication(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call("/v1/decompose", payload=request())
        self.assertEqual(caught.exception.code, 401)

    def test_decompose_endpoint_returns_validated_requirements(self):
        status, result = self.call(
            "/v1/decompose", key="client-secret", payload=request()
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["decomposition"]["status"], "DECOMPOSED")
        self.assertFalse(result["search_started"])

    def test_keyless_loopback_server_accepts_decomposition(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.server = build_server("127.0.0.1", 0, self.planner, "")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        status, result = self.call("/v1/decompose", payload=request())
        self.assertEqual(status, 200)
        self.assertEqual(result["decomposition"]["status"], "DECOMPOSED")


if __name__ == "__main__":
    unittest.main()
