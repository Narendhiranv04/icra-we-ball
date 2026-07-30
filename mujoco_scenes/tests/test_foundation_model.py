import json
import unittest
from unittest.mock import patch

from mujoco_scenes.foundation_model import (
    Candidate,
    FixedAssessmentBackend,
    FixedRankingBackend,
    FoundationModelResponseError,
    OpenAICompatibleRanker,
    RankingRequest,
    ServerConfig,
)


def _response(candidate_ids):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"ranked_candidate_ids": candidate_ids}
                    )
                }
            }
        ]
    }


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.path = None
        self.payload = None
        self.closed = False

    def post(self, path, payload):
        self.path = path
        self.payload = payload
        return self.response

    def close(self):
        self.closed = True


class FakeHttpResponse:
    status = 200
    reason = "OK"
    will_close = False

    def read(self):
        return json.dumps(
            _response(["spoon_1", "pen_1"])
        ).encode()


class FakeHttpConnection:
    def __init__(self):
        self.request_args = None
        self.closed = False

    def request(self, *args):
        self.request_args = args

    def getresponse(self):
        return FakeHttpResponse()

    def close(self):
        self.closed = True


class FoundationModelTests(unittest.TestCase):
    def setUp(self):
        self.candidates = (
            Candidate(
                "spoon_1",
                "spoon",
                {"visible": True, "insertable_in_mug": True},
            ),
            Candidate(
                "pen_1",
                "pen",
                {"visible": True, "insertable_in_mug": True},
            ),
        )
        self.request = RankingRequest(
            "can_stir",
            self.candidates,
            target={"id": "mug_1", "category": "mug"},
        )

    def test_valid_ranking_uses_only_supplied_observations(self):
        transport = FakeTransport(_response(["spoon_1", "pen_1"]))
        ranker = OpenAICompatibleRanker(
            ServerConfig(model="test-model"), transport
        )

        result = ranker.rank(self.request)

        self.assertEqual(result.candidate_ids, ("spoon_1", "pen_1"))
        self.assertEqual(result.model, "test-model")
        self.assertGreaterEqual(result.latency_ms, 0.0)
        self.assertEqual(transport.path, "/chat/completions")

        messages = transport.payload["messages"]
        visible_payload = json.loads(messages[1]["content"])
        self.assertEqual(
            [item["id"] for item in visible_payload["candidates"]],
            ["spoon_1", "pen_1"],
        )
        self.assertEqual(visible_payload["target"]["id"], "mug_1")
        schema = transport.payload["response_format"]["json_schema"]["schema"]
        item_schema = schema["properties"]["ranked_candidate_ids"]["items"]
        self.assertEqual(item_schema, {"type": "string"})

    def test_unobserved_candidate_is_rejected(self):
        transport = FakeTransport(_response(["spoon_1", "knife_1"]))
        ranker = OpenAICompatibleRanker(ServerConfig(), transport)

        with self.assertRaisesRegex(
            FoundationModelResponseError, "unobserved"
        ):
            ranker.rank(self.request)

    def test_incomplete_or_duplicate_ranking_is_rejected(self):
        for ranked_ids, message in (
            (["spoon_1"], "omitted"),
            (["spoon_1", "spoon_1"], "duplicate"),
        ):
            with self.subTest(ranked_ids=ranked_ids):
                ranker = OpenAICompatibleRanker(
                    ServerConfig(), FakeTransport(_response(ranked_ids))
                )
                with self.assertRaisesRegex(
                    FoundationModelResponseError, message
                ):
                    ranker.rank(self.request)

    def test_request_requires_unique_visible_candidates(self):
        duplicate = RankingRequest(
            "can_stir",
            (
                Candidate("spoon_1", "spoon"),
                Candidate("spoon_1", "spoon"),
            ),
        )
        ranker = OpenAICompatibleRanker(
            ServerConfig(), FakeTransport(_response([]))
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            ranker.rank(duplicate)

    def test_candidate_facts_must_be_json_serializable(self):
        request = RankingRequest(
            "can_stir",
            (Candidate("spoon_1", "spoon", {"bad": {1, 2}}),),
        )
        ranker = OpenAICompatibleRanker(
            ServerConfig(), FakeTransport(_response(["spoon_1"]))
        )

        with self.assertRaisesRegex(ValueError, "JSON"):
            ranker.rank(request)

    def test_fixed_backend_supports_offline_execution(self):
        result = FixedRankingBackend(["pen_1", "spoon_1"]).rank(
            self.request
        )

        self.assertEqual(result.candidate_ids, ("pen_1", "spoon_1"))
        self.assertEqual(result.model, "fixed")

    def test_functional_assessment_returns_a_visible_ranked_subset(self):
        transport = FakeTransport(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "functional_candidate_ids": [
                                        "spoon_1"
                                    ],
                                    "ranked_candidate_ids": ["spoon_1"],
                                }
                            )
                        }
                    }
                ]
            }
        )
        ranker = OpenAICompatibleRanker(ServerConfig(), transport)

        result = ranker.assess(self.request)

        self.assertEqual(
            result.functional_candidate_ids, ("spoon_1",)
        )
        self.assertEqual(result.ranked_candidate_ids, ("spoon_1",))

    def test_functional_assessment_rejects_unobserved_ids(self):
        backend = FixedAssessmentBackend(["knife_1"])

        with self.assertRaisesRegex(
            FoundationModelResponseError, "unobserved"
        ):
            backend.assess(self.request)

    def test_server_config_reads_environment_mapping(self):
        config = ServerConfig.from_env(
            {
                "TAMP_FM_BASE_URL": "https://models.example/v1",
                "TAMP_FM_MODEL": "ranker",
                "TAMP_FM_API_KEY": "secret",
                "TAMP_FM_TIMEOUT_SECONDS": "4.5",
                "TAMP_FM_MAX_TOKENS": "64",
            }
        )

        self.assertEqual(config.base_url, "https://models.example/v1")
        self.assertEqual(config.model, "ranker")
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.timeout_seconds, 4.5)
        self.assertEqual(config.max_tokens, 64)

    def test_context_manager_closes_transport(self):
        transport = FakeTransport(_response(["spoon_1", "pen_1"]))

        with OpenAICompatibleRanker(ServerConfig(), transport) as ranker:
            ranker.rank(self.request)

        self.assertTrue(transport.closed)

    def test_real_http_transport_uses_openai_endpoint_and_auth(self):
        connection = FakeHttpConnection()
        config = ServerConfig(
            base_url="http://models.example:8000/v1",
            api_key="test-key",
        )
        with patch(
            "mujoco_scenes.foundation_model.http.client.HTTPConnection",
            return_value=connection,
        ):
            with OpenAICompatibleRanker(config) as ranker:
                result = ranker.rank(self.request)

        self.assertEqual(result.candidate_ids, ("spoon_1", "pen_1"))
        method, path, _body, headers = connection.request_args
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertTrue(connection.closed)
